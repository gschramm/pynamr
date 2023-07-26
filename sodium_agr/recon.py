import argparse
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import h5py
import nibabel as nib
from pathlib import Path

import pymirc.viewer as pv

from preprocessing import TPIParameters
from reconstruction import channelwise_ifft_recon, channelwise_lsq_recon, regularized_sense_recon, dual_echo_sense_with_decay_estimation
from registration import align_images
from operators import projected_gradient_operator
from coils_sens import calculate_csm_inati_iter, calculate_sense_scale

#----------------------------------------------------------------
#----------------------------------------------------------------
#----------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--sdir', type=str, required=True)
parser.add_argument('--matrix_size',
                    type=int,
                    default=128,
                    help='recon matrix size, default 128')
parser.add_argument('--fov_cm',
                    type=float,
                    default=22.,
                    help='recon field of view (cm)')
parser.add_argument(
    '--beta_non_anatomical',
    type=float,
    default=1e-1,
    help='prior weight for recon with non-anatomical quadratic difference prior'
)
parser.add_argument('--beta_anatomical',
                    type=float,
                    default=3e-3,
                    help='prior weight for recon with anatomical L1 prior')
parser.add_argument(
    '--beta_r',
    type=float,
    default=3e-1,
    help='prior weight for anatomical L2 prior for the ratio image')

parser.add_argument('--max_iter_agr',
                    type=int,
                    default=500,
                    help='number of iterations for AGR without decay modeling')

parser.add_argument(
    '--max_outer_iter_agr',
    type=int,
    default=20,
    help='number of outer iterations for AGR with decay modeling')
args = parser.parse_args()

#---------------------------------------------------------------
#--- input parameters ------------------------------------------
#---------------------------------------------------------------

subject_path: Path = Path(args.sdir)
show_kspace_trajectory: bool = False

grid_shape = (args.matrix_size, args.matrix_size, args.matrix_size)
field_of_view_cm = args.fov_cm

beta_non_anatomical = args.beta_non_anatomical
beta_anatomical = args.beta_anatomical
beta_r = args.beta_r

max_iter_agr = args.max_iter_agr
max_outer_iter = args.max_outer_iter_agr

anat_path = subject_path / 'anatomical_prior_image.nii'
eta = 5e-3  # eta parameter for normalized joint gradient field of DTV operator

# echo times (ms) of first and 2nd echo
TE1_ms = 0.5
TE2_ms = 5.

#---------------------------------------------------------------
#--- create the output directory -------------------------------
#---------------------------------------------------------------

output_path = subject_path / f'recon_{"_".join(str(x) for x in grid_shape)}'
output_path.mkdir(exist_ok=True, parents=True)

# aff matrix for all recons
output_aff = np.diag(
    np.concatenate([10 * field_of_view_cm / np.array(grid_shape), [1]]))
#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

# load the multi-channel non-uniform k-space data
# the array will have shape (num_channels, num_points, num_readouts)
with h5py.File(subject_path / 'raw_TE05' / 'converted_data.h5', 'r') as f1:
    data_echo_1 = f1['data'][:]

    # read the GE's rotation tag
    if 'header' in f1:
        rotation = int(f1['header']['rdb_hdr']['rotation'][0])
    else:
        rotation = None

with h5py.File(subject_path / 'raw_TE5' / 'converted_data.h5', 'r') as f2:
    data_echo_2 = f2['data'][:]

if not np.iscomplexobj(data_echo_1):
    data_echo_1 = data_echo_1['real'] + 1j * data_echo_1['imag']

if not np.iscomplexobj(data_echo_2):
    data_echo_2 = data_echo_2['real'] + 1j * data_echo_2['imag']

# add a dummy coil axis, in case we have single coil 2D data
if data_echo_1.ndim == 2:
    data_echo_1 = np.expand_dims(data_echo_1, axis=0)

if data_echo_2.ndim == 2:
    data_echo_2 = np.expand_dims(data_echo_2, axis=0)

# normalize the data such that the maximum of the readouts of the first
# echo is approx. 1
# we do this to get a more consistent scaling of the the reconstruction

# we take the max of the 2nd point in each readout, since it is usually the max
data_norm = np.abs(data_echo_1[:, 1, :]).max()

data_echo_1 /= data_norm
data_echo_2 /= data_norm

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

# load the (pre-processed) kspace trajectory in 1/cm
with h5py.File(subject_path / 'kspace_trajectory.h5', 'r') as f:
    k = f['k'][...]
    g_params = TPIParameters(**f['k'].attrs)

# if GE's rotation tag is 0, we have to swap the x and y axis in the
# kspace trajectory to get the recons in RAS
if rotation == 0:
    tmp = k.copy()
    k[..., 0] = tmp[..., 1]
    k[..., 1] = -tmp[..., 0]
    del tmp

# ignore last data points in kspace trajectory
# (contains more points compared to data points)
k = k[:data_echo_1.shape[1], ...]

# show the k-space trajectory
if show_kspace_trajectory:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    step = 20
    for i in range(0, k.shape[1], 7):
        ax.scatter(k[::step, i, 0],
                   k[::step, i, 1],
                   k[::step, i, 2],
                   marker='.',
                   s=1)
    fig.show()

#---------------------------------------------------------------
#--- IFFT recons -----------------------------------------------
#---------------------------------------------------------------

ifft_file = output_path / 'coil_combined_ifft.npz'

if not ifft_file.exists():
    # calculate channel-wise IFFT
    iffts_1 = channelwise_ifft_recon(data_echo_1,
                                     k,
                                     field_of_view_cm=field_of_view_cm,
                                     width=2)
    iffts_2 = channelwise_ifft_recon(data_echo_2,
                                     k,
                                     field_of_view_cm=field_of_view_cm,
                                     width=2)

    _, coil_combined_ifft_1 = calculate_csm_inati_iter(iffts_1, smoothing=7)
    _, coil_combined_ifft_2 = calculate_csm_inati_iter(iffts_2, smoothing=7)

    # save the IFFTs
    ifft_aff = np.diag(
        np.concatenate(
            [10 * field_of_view_cm / np.array(iffts_1.shape[1:]), [1]]))

    print(f'saving {ifft_file}')
    np.savez_compressed(ifft_file,
                        coil_combined_ifft_1=coil_combined_ifft_1,
                        coil_combined_ifft_2=coil_combined_ifft_2,
                        ifft_aff=ifft_aff)
else:
    print(f'loading {ifft_file}')
    data = np.load(ifft_file)
    coil_combined_ifft_1 = data['coil_combined_ifft_1']
    coil_combined_ifft_2 = data['coil_combined_ifft_2']
    iff_aff = data['ifft_aff']

#---------------------------------------------------------------
#--- early stopped channel-wise LSQ recons for coil sens -------
#---------------------------------------------------------------

lsq_early_file = output_path / 'channelwise_lsq_early.npz'

if not lsq_early_file.exists():
    # early stopped least squares recons
    lsq_1 = channelwise_lsq_recon(data_echo_1,
                                  k,
                                  field_of_view_cm=field_of_view_cm,
                                  grid_shape=grid_shape)
    lsq_2 = channelwise_lsq_recon(data_echo_2,
                                  k,
                                  field_of_view_cm=field_of_view_cm,
                                  grid_shape=grid_shape)

    print(f'saving {lsq_early_file}')
    np.savez(lsq_early_file, lsq_1=lsq_1, lsq_2=lsq_2)
else:
    print(f'loading {lsq_early_file}')
    data = np.load(lsq_early_file)
    lsq_1 = data['lsq_1']
    lsq_2 = data['lsq_2']

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

coil_sens_file = output_path / 'coil_sens_maps.npz'

if not coil_sens_file.exists():
    # calculate the coil sensitivities
    if data_echo_1.shape[0] > 1:
        coil_sens_maps_1, coil_combined_lsq_1 = calculate_csm_inati_iter(
            lsq_1, smoothing=7)
        coil_sens_maps_2, coil_combined_lsq_2 = calculate_csm_inati_iter(
            lsq_2, smoothing=7)
    else:
        coil_sens_maps_1 = np.ones(lsq_1.shape, dtype=lsq_1.dtype)
        coil_sens_maps_2 = np.ones(lsq_2.shape, dtype=lsq_2.dtype)
        coil_combined_lsq_1 = lsq_1[0, ...]
        coil_combined_lsq_2 = lsq_2[0, ...]

    # calculate a scaling factor for the sensitivities such that the resulting
    # forward operator has norm 1
    coil_sens_scale = calculate_sense_scale(coil_sens_maps_1, k, data_echo_1,
                                            field_of_view_cm)

    coil_sens_maps_1 /= coil_sens_scale
    coil_sens_maps_2 /= coil_sens_scale

    # apply the sense scaling to the sos_lsq image
    coil_combined_lsq_1 *= coil_sens_scale
    coil_combined_lsq_2 *= coil_sens_scale

    coil_sens_maps = coil_sens_maps_1

    # save the coil sensitivities
    print(f'saving {coil_sens_file}')
    np.savez_compressed(coil_sens_file,
                        coil_sens_maps=coil_sens_maps,
                        coil_combined_lsq_1=coil_combined_lsq_1,
                        coil_combined_lsq_2=coil_combined_lsq_2,
                        coil_sens_scale=coil_sens_scale,
                        output_aff=output_aff)
else:
    print(f'loading {coil_sens_file}')
    data = np.load(coil_sens_file)
    coil_sens_maps = data['coil_sens_maps']
    coil_combined_lsq_1 = data['coil_combined_lsq_1']
    coil_combined_lsq_2 = data['coil_combined_lsq_2']
    coil_sens_scale = data['coil_sens_scale']
    output_aff = data['output_aff']

#---------------------------------------------------------------
#--- SENSE recons with non-anatomical quad diff prior ----------
#---------------------------------------------------------------
# regularized sense recons

non_agr_sens_file_1 = output_path / f'sense_non_agr_b{beta_non_anatomical:.2E}_1.npy'
non_agr_sens_file_2 = output_path / f'sense_non_agr_b{beta_non_anatomical:.2E}_2.npy'

if not non_agr_sens_file_1.exists():
    x0 = cp.asarray(coil_combined_lsq_1.astype(data_echo_1.dtype))
    sense_L2_1, _ = regularized_sense_recon(data_echo_1,
                                            coil_sens_maps,
                                            k,
                                            beta=beta_non_anatomical,
                                            regularization='L2',
                                            field_of_view_cm=field_of_view_cm,
                                            x=x0)
    print(f'saving {non_agr_sens_file_1}')
    np.save(non_agr_sens_file_1, sense_L2_1)
else:
    print(f'loading {non_agr_sens_file_1}')
    sense_L2_1 = np.load(non_agr_sens_file_1)

if not non_agr_sens_file_2.exists():
    x0 = cp.asarray(coil_combined_lsq_2.astype(data_echo_2.dtype))
    sense_L2_2, _ = regularized_sense_recon(data_echo_2,
                                            coil_sens_maps,
                                            k,
                                            beta=beta_non_anatomical,
                                            regularization='L2',
                                            field_of_view_cm=field_of_view_cm,
                                            x=x0)

    print(f'saving {non_agr_sens_file_2}')
    np.save(non_agr_sens_file_2, sense_L2_2)
else:
    print(f'loading {non_agr_sens_file_2}')
    sense_L2_2 = np.load(non_agr_sens_file_2)

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

# align the anatomical prior image and setup the projected gradient operator
anat_aligned_path = output_path / f'{anat_path.stem}_aligned.nii'

if not anat_aligned_path.exists():
    anat_nii = nib.as_closest_canonical(nib.load(anat_path))
    anat_img = anat_nii.get_fdata()
    anat_img /= np.percentile(anat_img, 99.9)

    anat_voxsize = anat_nii.header['pixdim'][1:4]
    anat_origin = anat_nii.affine[:-1, -1]

    na_img = np.abs(sense_L2_1)
    na_voxsize = 10 * field_of_view_cm / np.array(grid_shape)
    na_origin = anat_origin.copy()

    anat_img_aligned, transform = align_images(na_img, anat_img, na_voxsize,
                                               na_origin, anat_voxsize,
                                               anat_origin)

    # save the aligned anat image
    print(f'saving {anat_aligned_path}')
    nib.save(nib.Nifti1Image(anat_img_aligned, output_aff), anat_aligned_path)
else:
    print(f'loading {anat_aligned_path}')
    anat_img_aligned = nib.as_closest_canonical(
        nib.load(anat_aligned_path)).get_fdata()

# setup the projected gradient operator we need for AGR with DTV
PG = projected_gradient_operator(cp, anat_img_aligned, eta=eta)

#---------------------------------------------------------------
#--- AGR with DTV prior ----------------------------------------
#---------------------------------------------------------------

agr_nodecay_file_1 = output_path / f'agr_no_decay_b{beta_anatomical:.2E}_1.npz'
agr_nodecay_file_2 = output_path / f'agr_no_decay_b{beta_anatomical:.2E}_2.npz'

if not agr_nodecay_file_1.exists():
    agr_L1_1, u_agr_L1_1 = regularized_sense_recon(
        data_echo_1,
        coil_sens_maps,
        k,
        beta=beta_anatomical,
        regularization='L1',
        G=PG,
        field_of_view_cm=field_of_view_cm,
        x=cp.asarray(sense_L2_1),
        max_iter=max_iter_agr)

    print(f'saving {agr_nodecay_file_1}')
    np.savez_compressed(agr_nodecay_file_1,
                        agr_L1_1=agr_L1_1,
                        u_agr_L1_1=u_agr_L1_1)
    nib.save(nib.Nifti1Image(np.abs(agr_L1_1), output_aff),
             agr_nodecay_file_1.with_suffix('.nii'))
else:
    print(f'loading {agr_nodecay_file_1}')
    data = np.load(agr_nodecay_file_1)
    agr_L1_1 = data['agr_L1_1']
    u_agr_L1_1 = data['u_agr_L1_1']

if not agr_nodecay_file_2.exists():
    agr_L1_2, u_agr_L1_2 = regularized_sense_recon(
        data_echo_2,
        coil_sens_maps,
        k,
        beta=beta_anatomical,
        regularization='L1',
        G=PG,
        field_of_view_cm=field_of_view_cm,
        x=cp.asarray(sense_L2_1),
        max_iter=max_iter_agr)

    print(f'saving {agr_nodecay_file_2}')
    np.savez_compressed(agr_nodecay_file_2,
                        agr_L1_2=agr_L1_2,
                        u_agr_L1_2=u_agr_L1_2)
    nib.save(nib.Nifti1Image(np.abs(agr_L1_2), output_aff),
             agr_nodecay_file_2.with_suffix('.nii'))
else:
    print(f'loading {agr_nodecay_file_2}')
    data = np.load(agr_nodecay_file_2)
    agr_L1_2 = data['agr_L1_2']
    u_agr_L1_2 = data['u_agr_L1_2']

#---------------------------------------------------------------
#--- dual echo AGR with decay est ------------------------------
#---------------------------------------------------------------

# estimate the ratio between the two AGR recons
r0 = np.clip(np.abs(agr_L1_2) / np.abs(agr_L1_1), 0, 1)
# set ratio to one in voxels where there is low signal in the first echo
mask = 1 - (np.abs(agr_L1_1) < 0.05 * np.abs(agr_L1_1).max())

label, num_label = ndimage.label(mask == 1)
size = np.bincount(label.ravel())
biggest_label = size[1:].argmax() + 1
clump_mask = (label == biggest_label)

r0[clump_mask == 0] = 1

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

dual_echo_agr_file = output_path / f'dual_echo_agr_w_decay_b{beta_anatomical:.2E}_br{beta_r:.2E}_n{max_outer_iter}.npz'

if not dual_echo_agr_file.exists():
    num_data = np.size(data_echo_1)

    # initialize the dual variable for PDHG
    u0 = np.concatenate(
        (u_agr_L1_1[:num_data], u_agr_L1_2[:num_data], u_agr_L1_1[num_data:]))

    dual_echo_agr_w_decay, r = dual_echo_sense_with_decay_estimation(
        data_echo_1,
        data_echo_2,
        g_params.sampling_time_us,
        TE1_ms,
        TE2_ms,
        coil_sens_maps,
        k,
        x0=agr_L1_1,
        u0=u0,
        r0=r0,
        G=PG,
        field_of_view_cm=field_of_view_cm,
        regularization='L1',
        beta=beta_anatomical,
        max_iter=100,
        max_outer_iter=max_outer_iter,
        num_time_bins=64,
        beta_r=beta_r)

    print(f'saving {dual_echo_agr_file}')
    np.savez_compressed(dual_echo_agr_file,
                        dual_echo_agr_w_decay=dual_echo_agr_w_decay,
                        r=r)
    nib.save(nib.Nifti1Image(np.abs(dual_echo_agr_w_decay), output_aff),
             dual_echo_agr_file.with_suffix('.nii'))
    nib.save(nib.Nifti1Image(np.abs(r), output_aff),
             dual_echo_agr_file.parent / f'{dual_echo_agr_file.stem}_r.nii')
else:
    print(f'loading {dual_echo_agr_file}')
    data = np.load(dual_echo_agr_file)
    dual_echo_agr_w_decay = data['dual_echo_agr_w_decay']
    r = data['r']
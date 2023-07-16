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
args = parser.parse_args()

subject_path: Path = Path(args.sdir)
show_kspace_trajectory: bool = False

grid_shape = (128, 128, 128)
#grid_shape = (190, 190, 190)
field_of_view_cm = 22.

beta_non_anatomical = 1e-1
beta_anatomical = 1e-3
beta_r = 3e-1

max_iter_agr = 500
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
    np.concatenate([10 * field_of_view_cm / np.array(iffts_1.shape[1:]), [1]]))
nib.save(nib.Nifti1Image(np.abs(coil_combined_ifft_1), ifft_aff),
         output_path / 'coil_combined_ifft_1.nii')
nib.save(nib.Nifti1Image(np.abs(coil_combined_ifft_2), ifft_aff),
         output_path / 'coil_combined_ifft_2.nii')

#---------------------------------------------------------------
#--- early stopped channel-wise LSQ recons for coil sens -------
#---------------------------------------------------------------

# early stopped least squares recons
lsq_1 = channelwise_lsq_recon(data_echo_1,
                              k,
                              field_of_view_cm=field_of_view_cm,
                              grid_shape=grid_shape)
lsq_2 = channelwise_lsq_recon(data_echo_2,
                              k,
                              field_of_view_cm=field_of_view_cm,
                              grid_shape=grid_shape)

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

# calculate the coil sensitivities
coil_sens_maps_1, coil_combined_lsq_1 = calculate_csm_inati_iter(lsq_1,
                                                                 smoothing=7)
coil_sens_maps_2, coil_combined_lsq_2 = calculate_csm_inati_iter(lsq_2,
                                                                 smoothing=7)

coil_sens_scale = calculate_sense_scale(coil_sens_maps_1, k, data_echo_1,
                                        field_of_view_cm)

coil_sens_maps_1 /= coil_sens_scale
coil_sens_maps_2 /= coil_sens_scale

# apply the sense scaling to the sos_lsq image
coil_combined_lsq_1 *= coil_sens_scale
coil_combined_lsq_2 *= coil_sens_scale

coil_sens_maps = coil_sens_maps_1

# save the coil sensitivities
nib.save(nib.Nifti1Image(coil_sens_maps, output_aff),
         output_path / 'coil_sens_maps.nii')
nib.save(nib.Nifti1Image(np.abs(coil_combined_lsq_1), output_aff),
         output_path / 'early_lsq_1.nii')
nib.save(nib.Nifti1Image(np.abs(coil_combined_lsq_2), output_aff),
         output_path / 'early_lsq_2.nii')
#---------------------------------------------------------------
#--- SENSE recons with non-anatomical quad diff prior ----------
#---------------------------------------------------------------

# regularized sense recons
x0 = cp.asarray(coil_combined_lsq_1.astype(data_echo_1.dtype))
sense_L2_1, _ = regularized_sense_recon(data_echo_1,
                                        coil_sens_maps,
                                        k,
                                        beta=beta_non_anatomical,
                                        regularization='L2',
                                        field_of_view_cm=field_of_view_cm,
                                        x=x0)

x0 = cp.asarray(coil_combined_lsq_2.astype(data_echo_2.dtype))
sense_L2_2, _ = regularized_sense_recon(data_echo_2,
                                        coil_sens_maps,
                                        k,
                                        beta=beta_non_anatomical,
                                        regularization='L2',
                                        field_of_view_cm=field_of_view_cm,
                                        x=x0)

# save the sense recons with quad diff prior
nib.save(nib.Nifti1Image(sense_L2_1, output_aff),
         output_path / f'sense_quad_prior_b{beta_anatomical:.2E}_1.nii')
nib.save(nib.Nifti1Image(sense_L2_2, output_aff),
         output_path / f'sense_quad_prior_b{beta_anatomical:.2E}_2.nii')
#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

# align the anatomical prior image and setup the projected gradient operator

anat_path = subject_path / 'anatomical_prior_image.nii'

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
nib.save(nib.Nifti1Image(anat_img_aligned, output_aff),
         output_path / f'anatomical_prior_image_aligned.nii')

# setup the projected gradient operator we need for AGR with DTV
PG = projected_gradient_operator(cp, anat_img_aligned, eta=5e-3)

#---------------------------------------------------------------
#--- AGR with DTV prior ----------------------------------------
#---------------------------------------------------------------

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

# save the AGR recons
nib.save(nib.Nifti1Image(agr_L1_1, output_aff),
         output_path / f'agr_dtv_b{beta_anatomical:.2E}_1.nii')
np.save(output_path / f'u_agr_dtv_b{beta_anatomical:.2E}_1.npy', u_agr_L1_1)

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

# save the AGR recons
nib.save(nib.Nifti1Image(agr_L1_2, output_aff),
         output_path / f'agr_dtv_b{beta_anatomical:.2E}_2.nii')
np.save(output_path / f'u_agr_dtv_b{beta_anatomical:.2E}_2.npy', u_agr_L1_2)

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

TE1_ms = 0.5
TE2_ms = 5.

num_data = np.prod(data_echo_1.shape)
u0 = np.concatenate(
    (u_agr_L1_1[:num_data], u_agr_L1_2[:num_data], u_agr_L1_1[num_data:]))

a, b = dual_echo_sense_with_decay_estimation(data_echo_1,
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
                                             max_outer_iter=20,
                                             num_time_bins=64,
                                             beta_r=beta_r)

#---------------------------------------------------------------
#--- show results ----------------------------------------------
#---------------------------------------------------------------

vi = pv.ThreeAxisViewer(
    [np.abs(sense_L2_1),
     np.abs(agr_L1_1), anat_img_aligned],
    imshow_kwargs=dict(cmap='Greys_r', vmax=1))

vi.fig.savefig(output_path / 'recons.png')
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import h5py
import nibabel as nib
from pathlib import Path

import sigpy
import sigpy.mri

import pymirc.viewer as pv

from preprocessing import TPIParameters
from reconstruction import channelwise_ifft_recon, channelwise_lsq_recon, regularized_sense_recon
from registration import align_images
from operators import projected_gradient_operator

#----------------------------------------------------------------
#----------------------------------------------------------------
#----------------------------------------------------------------

subject_path: Path = Path('/data/sodium_mr/sodium_data/CSF-032')
show_kspace_trajectory: bool = False

grid_shape = (128, 128, 128)
field_of_view_cm = 22.

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

# load the multi-channel non-uniform k-space data
# the array will have shape (num_channels, num_points, num_readouts)
with h5py.File(subject_path / 'raw_TE05' / 'converted_data.h5', 'r') as f1:
    data_echo_1 = f1['data'][:]

with h5py.File(subject_path / 'raw_TE5' / 'converted_data.h5', 'r') as f2:
    data_echo_2 = f2['data'][:]

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
#---------------------------------------------------------------
#---------------------------------------------------------------

# calculate channel-wise IFFT
iffts_1 = channelwise_ifft_recon(data_echo_1,
                                 k,
                                 field_of_view_cm=field_of_view_cm)
iffts_2 = channelwise_ifft_recon(data_echo_2,
                                 k,
                                 field_of_view_cm=field_of_view_cm)

sos_ifft_1 = ((np.abs(iffts_1)**2).sum(axis=0))**0.5
sos_ifft_2 = ((np.abs(iffts_2)**2).sum(axis=0))**0.5

# early stopped least squares recons
lsq_1 = channelwise_lsq_recon(data_echo_1,
                              k,
                              field_of_view_cm=field_of_view_cm,
                              grid_shape=grid_shape)
lsq_2 = channelwise_lsq_recon(data_echo_2,
                              k,
                              field_of_view_cm=field_of_view_cm,
                              grid_shape=grid_shape)

sos_lsq_1 = ((np.abs(lsq_1)**2).sum(axis=0))**0.5
sos_lsq_2 = ((np.abs(lsq_2)**2).sum(axis=0))**0.5

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

# calculate the coil sensitivities
sm_sig = 1.5 * grid_shape[0] / 128
sos_lsq_1_sm = gaussian_filter(sos_lsq_1, sm_sig)
sos_lsq_2_sm = gaussian_filter(sos_lsq_2, sm_sig)

sens_1 = np.array([gaussian_filter(x, sm_sig) / sos_lsq_1_sm for x in lsq_1])
sens_2 = np.array([gaussian_filter(x, sm_sig) / sos_lsq_2_sm for x in lsq_2])

# we normalize the sensitivities such that the sens operator has norm one
# this is important when doing sense recons with 2nd (gradient) operator
# and using PDHG
k_d = cp.asarray(k.reshape(-1, 3)) * field_of_view_cm
d_d = cp.asarray(data_echo_1.reshape(data_echo_1.shape[0], -1))
S_tmp = sigpy.mri.linop.Sense(cp.asarray(sens_1), coord=k_d)
max_eig = sigpy.app.MaxEig(S_tmp.H * S_tmp, dtype=d_d.dtype,
                           device=k_d.device).run()

sens_scale = np.sqrt(max_eig)

sens_1 /= sens_scale
sens_2 /= sens_scale

# apply the sense scaling to the sos_lsq image

sos_lsq_1 *= sens_scale
sos_lsq_2 *= sens_scale

sos_lsq_1_sm *= sens_scale
sos_lsq_2_sm *= sens_scale

coil_sens = sens_1
#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

beta = 1e-1

# regularized sense recons
x0 = cp.asarray(sos_lsq_1_sm.astype(data_echo_1.dtype))
sense_L2_1 = regularized_sense_recon(data_echo_1,
                                     coil_sens,
                                     k,
                                     beta=beta,
                                     regulariztion='L2',
                                     field_of_view_cm=field_of_view_cm,
                                     x=x0)

x0 = cp.asarray(sos_lsq_1_sm.astype(data_echo_2.dtype))
sense_L2_2 = regularized_sense_recon(data_echo_2,
                                     coil_sens,
                                     k,
                                     beta=beta,
                                     regulariztion='L2',
                                     field_of_view_cm=field_of_view_cm,
                                     x=x0)

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

PG = projected_gradient_operator(cp, anat_img_aligned, eta=5e-3)

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

beta = 3e-3
max_iter = 500

agr_L1_1 = regularized_sense_recon(data_echo_1,
                                   coil_sens,
                                   k,
                                   beta=beta,
                                   regulariztion='L1',
                                   G=PG,
                                   field_of_view_cm=field_of_view_cm,
                                   x=cp.asarray(sense_L2_1),
                                   max_iter=max_iter)
np.save(subject_path / 'agr_L1_1.npy', agr_L1_1)

agr_L1_2 = regularized_sense_recon(data_echo_2,
                                   coil_sens,
                                   k,
                                   beta=beta,
                                   regulariztion='L1',
                                   G=PG,
                                   field_of_view_cm=field_of_view_cm,
                                   x=cp.asarray(sense_L2_1),
                                   max_iter=max_iter)
np.save(subject_path / 'agr_L1_2.npy', agr_L1_2)

vi = pv.ThreeAxisViewer(
    [np.abs(agr_L1_2), np.abs(agr_L1_1), anat_img_aligned],
    imshow_kwargs=dict(cmap='Greys_r', vmax=1))

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import h5py
from pathlib import Path

import sigpy
import sigpy.mri

import pymirc.viewer as pv

from preprocessing import TPIParameters
from reconstruction import channelwise_ifft_recon, channelwise_lsq_recon, regularized_sense_recon

#----------------------------------------------------------------
#----------------------------------------------------------------
#----------------------------------------------------------------

subject_path: Path = Path('/data/sodium_mr/sodium_data/EP-005')
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

beta = 1e-3

# regularized sense recons
x0 = cp.asarray(sos_lsq_1_sm.astype(data_echo_1.dtype))
sense_L1_1 = regularized_sense_recon(data_echo_1,
                                     coil_sens,
                                     k,
                                     beta=beta,
                                     regulariztion='L1',
                                     field_of_view_cm=field_of_view_cm,
                                     x=x0)

x0 = cp.asarray(sos_lsq_1_sm.astype(data_echo_2.dtype))
sense_L1_2 = regularized_sense_recon(data_echo_2,
                                     coil_sens,
                                     k,
                                     beta=beta,
                                     regulariztion='L1',
                                     field_of_view_cm=field_of_view_cm,
                                     x=x0)

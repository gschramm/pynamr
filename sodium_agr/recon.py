import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

import sigpy
from preprocessing import TPIParameters

from scipy.ndimage import gaussian_filter
import pymirc.viewer as pv


def ifft_recon(data: np.ndarray,
               k: np.ndarray,
               grid_shape=(64, 64, 64),
               field_of_view_cm=22.,
               **kwargs) -> np.ndarray:

    # transfer k-space trajectory to GPU and convert to unitless 
    k_d = cp.asarray(k.reshape(-1, 3)) * field_of_view_cm

    samp_dens = sigpy.gridding(cp.ones(k_d.shape[0], dtype=data.dtype), k_d,
                               grid_shape, **kwargs)
    kernel_gridded = sigpy.gridding(cp.ones(1, dtype=data.dtype),
                                    cp.zeros((1, 3), dtype=k_d.dtype),
                                    grid_shape, **kwargs)

    ifft_op = sigpy.linop.IFFT(grid_shape)
    kernel_ifft_d = ifft_op(kernel_gridded)

    x_ifft_d = cp.zeros((data.shape[0], ) + grid_shape, dtype=data.dtype)

    for i, d in enumerate(data):
        # transfer data to GPU
        d_d = cp.asarray(d.ravel())

        data_gridded = sigpy.gridding(d_d, k_d, grid_shape, **kwargs)

        data_gridded_corr = data_gridded.copy()
        data_gridded_corr[samp_dens > 0] /= samp_dens[samp_dens > 0]

        x_ifft_d[i, ...] = ifft_op(data_gridded_corr) / kernel_ifft_d

    return cp.asnumpy(x_ifft_d)


#----------------------------------------------------------------
#----------------------------------------------------------------
#----------------------------------------------------------------

subject_path: Path = Path('/data/sodium_mr/sodium_data/EP-005')
show_kspace_trajectory: bool = False

ishape = (64, 64, 64)
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
iffts_1 = ifft_recon(data_echo_1, k)
iffts_2 = ifft_recon(data_echo_2, k)

#k_d = cp.asarray(k.reshape(-1, 3))
#data_d = cp.asanyarray(data_echo_2[0, ...]).ravel()
#
## quick iterative LSQ recon
#A = sigpy.linop.NUFFT(ishape, k_d)
#alg = sigpy.app.LinearLeastSquares(A, data_d, max_iter=20)
#x = cp.asnumpy(alg.run())

#
#x_ifft = np.array(x_ifft)
#x_lsq = np.array(x_lsq)
#
#sos_ifft = ((np.abs(x_ifft)**2).sum(axis=0))**0.5
#sos_lsq = ((np.abs(x_lsq)**2).sum(axis=0))**0.5
#
##vi = pv.ThreeAxisViewer(
##    [gaussian_filter(np.abs(x_ifft), (0, .75, .75, .75)),
##     np.abs(x_ref)],
##    imshow_kwargs=dict(cmap='Greys_r'))
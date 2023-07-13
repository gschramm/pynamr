import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import h5py
from pathlib import Path

import sigpy
import sigpy.mri

from preprocessing import TPIParameters
import pymirc.viewer as pv


def channelwise_ifft_recon(data: np.ndarray,
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


def channelwise_lsq_recon(data: np.ndarray,
                          k: np.ndarray,
                          grid_shape=(64, 64, 64),
                          field_of_view_cm=22.,
                          max_iter=10,
                          **kwargs) -> np.ndarray:

    # transfer k-space trajectory to GPU and convert to unitless
    k_d = cp.asarray(k.reshape(-1, 3)) * field_of_view_cm
    A = sigpy.linop.NUFFT(grid_shape, k_d)

    x = cp.zeros((data.shape[0], ) + grid_shape, dtype=data.dtype)

    for i, d in enumerate(data):
        # transfer data to GPU
        d_d = cp.asarray(d.ravel())
        alg = sigpy.app.LinearLeastSquares(A, d_d, max_iter=max_iter, **kwargs)
        x[i, ...] = alg.run()

    return cp.asnumpy(x)


#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------


def regularized_sense_recon(data: np.ndarray,
                            coil_sens: np.ndarray,
                            k: np.ndarray,
                            field_of_view_cm=22.,
                            regulariztion: str = 'L2',
                            beta: float = 1e-1,
                            sigma: float = 1e-1,
                            operator_norm_squared: float | None = 1.,
                            max_iter: int = 100,
                            G: sigpy.linop.Linop | None = None,
                            **kwargs) -> np.ndarray:

    # send kspace trajectory and data to GPU
    k_d = cp.asarray(k.reshape(-1, 3)) * field_of_view_cm
    d_d = cp.asarray(data.reshape(data.shape[0], -1))

    # setup normalize Sense operator
    S = sigpy.mri.linop.Sense(cp.asarray(coil_sens), coord=k_d)

    # setup normalized gradient operator
    if G is None:
        G = (1 / np.sqrt(12)) * sigpy.linop.FiniteDifference(S.ishape)

    # setup prox for gradient norm
    if regulariztion == 'L2':
        proxg = sigpy.prox.L2Reg(G.oshape, lamda=beta)
        g = lambda z: float(beta * 0.5 * ((z * z.conj()).sum()).real)
    elif regulariztion == 'L1':
        proxg = sigpy.prox.L1Reg(G.oshape, lamda=beta)
        g = lambda z: float(beta * cp.abs(z).sum())
    else:
        raise ValueError('regularization must be L1 or L2')

    # setup initial guess
    x0 = cp.asarray(sos_lsq_1_sm.astype(np.complex64))

    if operator_norm_squared is None:
        operator_norm_squared = sigpy.app.MaxEig(S.H * S,
                                                 dtype=data.dtype,
                                                 device=data.device).run()

    alg = sigpy.app.LinearLeastSquares(S,
                                       d_d,
                                       G=G,
                                       proxg=proxg,
                                       x=x0,
                                       max_iter=max_iter,
                                       sigma=sigma,
                                       tau=0.99 * operator_norm_squared /
                                       sigma,
                                       g=g,
                                       **kwargs)
    x = cp.asnumpy(alg.run())

    return x


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
iffts_1 = channelwise_ifft_recon(data_echo_1, k)
iffts_2 = channelwise_ifft_recon(data_echo_2, k)

sos_ifft_1 = ((np.abs(iffts_1)**2).sum(axis=0))**0.5
sos_ifft_2 = ((np.abs(iffts_2)**2).sum(axis=0))**0.5

# early stopped least squares recons
lsq_1 = channelwise_lsq_recon(data_echo_1, k)
lsq_2 = channelwise_lsq_recon(data_echo_2, k)

sos_lsq_1 = ((np.abs(lsq_1)**2).sum(axis=0))**0.5
sos_lsq_2 = ((np.abs(lsq_2)**2).sum(axis=0))**0.5

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

# calculate the coil sensitivities
sigma = 1.5
sos_lsq_1_sm = gaussian_filter(sos_lsq_1, sigma)
sos_lsq_2_sm = gaussian_filter(sos_lsq_2, sigma)

sens_1 = np.array([gaussian_filter(x, sigma) / sos_lsq_1_sm for x in lsq_1])
sens_2 = np.array([gaussian_filter(x, sigma) / sos_lsq_2_sm for x in lsq_2])

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

#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------

beta = 1e-3

# regularized sense recons
sense_L1_1 = regularized_sense_recon(data_echo_1,
                                     sens_1,
                                     k,
                                     beta=beta,
                                     regulariztion='L1')

sense_L1_2 = regularized_sense_recon(data_echo_2,
                                     sens_2,
                                     k,
                                     beta=beta,
                                     regulariztion='L1')

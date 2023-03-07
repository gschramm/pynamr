import h5py
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import sigpy
import pymirc.viewer as pv

from copy import deepcopy
from pathlib import Path

# input parameters

gradient_file: str = '/data/sodium_mr/tpi_gradients/ak_grad56.h5'
echo_1_data_file: str = '/data/sodium_mr/20230225_MR3_GS_TPI/pfiles/P51200.7.h5'
echo_2_data_file: str = '/data/sodium_mr/20230225_MR3_GS_TPI/pfiles/P52224.7.h5'

field_of_view_cm: float = 22.
ishape = (128, 128, 128)

regularization_norm_non_anatomical = 'L2'
beta_non_anatomical = 5e-2  # ca 1e-3 for L1, 5e-2 for L2

sigma = 0.1
max_num_iter = 200

data_scale = 1e-8
nufft_scale = 0.00884

odir = Path(echo_1_data_file).parents[1] / 'recons'
odir.mkdir(exist_ok=True, parents=True)

#--------------------------------------------------------------------
#--------------------------------------------------------------------
# read the gradient file to get the k-space trajectory
#--------------------------------------------------------------------
#--------------------------------------------------------------------

with h5py.File(gradient_file) as data:
    # gradients come in T/m
    grads_T_m = np.transpose(data['/gradients'][:], (1, 2, 0))

# time sampling step in micro seconds
dt_us = 16.

# gamma by 2pi in MHz/T
gamma_by_2pi_MHz_T: float = 11.262

# k values in 1/cm with shape (num_readouts, num_samples_per_readout, 3)
k_array = 0.01 * np.cumsum(grads_T_m, axis=1) * dt_us * gamma_by_2pi_MHz_T

# sigpy needs the k-space cooridnates without units
# -> we have to multiply by the field of view in cm

k_array *= field_of_view_cm

# send k_array to GPU
k_array = cp.asarray(k_array)

#--------------------------------------------------------------------
#--------------------------------------------------------------------
# read the acquired data
#--------------------------------------------------------------------
#--------------------------------------------------------------------

with h5py.File(echo_1_data_file) as data1:
    data_echo_1 = data1['/data'][:].T

with h5py.File(echo_2_data_file) as data2:
    data_echo_2 = data2['/data'][:].T

# convert data to complex (from two reals) send data arrays to GPU
data_echo_1 = cp.asarray(data_echo_1['real'] + 1j * data_echo_1['imag'])
data_echo_2 = cp.asarray(data_echo_2['real'] + 1j * data_echo_2['imag'])

# scale the data such that we get CSF approx 3 with normalized nufft operator
data_echo_1 *= data_scale
data_echo_2 *= data_scale

## plot the data
#ims1 = dict(vmin=0, vmax=cp.abs(data_echo_1).max(), aspect=1 / 25)
#ims2 = dict(vmin=-np.pi, vmax=np.pi, aspect=1 / 25., cmap='seismic')
#
#num_points = 60
#sample_point = 20
#
#fig, ax = plt.subplots(2, 4, figsize=(16, 8))
#im00 = ax[0, 0].imshow(np.abs(cp.asnumpy(data_echo_1[:, :num_points])), **ims1)
#im10 = ax[1, 0].imshow(np.abs(cp.asnumpy(data_echo_2[:, :num_points])), **ims1)
#im01 = ax[0, 1].imshow(np.angle(cp.asnumpy(data_echo_1[:, :num_points])),
#                       **ims2)
#im11 = ax[1, 1].imshow(np.angle(cp.asnumpy(data_echo_2[:, :num_points])),
#                       **ims2)
#
#ax[0, 2].plot(np.abs(cp.asnumpy(data_echo_1[:4, :num_points])).T, '.-', lw=0.5)
#ax[1, 2].plot(np.abs(cp.asnumpy(data_echo_2[:4, :num_points])).T, '.-', lw=0.5)
#
#ax[0, 3].plot(np.abs(cp.asnumpy(data_echo_1[:, sample_point])), '.-', lw=0.5)
#ax[1, 3].plot(np.abs(cp.asnumpy(data_echo_2[:, sample_point])), '.-', lw=0.5)
#
#ax[0, 0].set_title(f'abs first {num_points} samples')
#ax[0, 1].set_title('phase')
#ax[0, 2].set_title('abs vs readout time (hor. profile)')
#ax[0, 3].set_title('abs vs readout (vert. profile)')
#ax[0, 0].set_ylabel('echo 1')
#ax[1, 0].set_ylabel('echo 2')
#
#fig.colorbar(im00, ax=ax[0, 0], location='bottom', fraction=0.04)
#fig.colorbar(im10, ax=ax[1, 0], location='bottom', fraction=0.04)
#fig.colorbar(im01, ax=ax[0, 1], location='bottom', fraction=0.04)
#fig.colorbar(im11, ax=ax[1, 1], location='bottom', fraction=0.04)
#
#fig.tight_layout()
#fig.show()

#--------------------------------------------------------------------
#--------------------------------------------------------------------
# regrid the data and do simple IFFT
#--------------------------------------------------------------------
#--------------------------------------------------------------------

data_echo_1_gridded = sigpy.gridding(data_echo_1.ravel(),
                                     k_array.reshape(-1, 3),
                                     ishape,
                                     kernel='spline',
                                     width=1)
data_echo_2_gridded = sigpy.gridding(data_echo_2.ravel(),
                                     k_array.reshape(-1, 3),
                                     ishape,
                                     kernel='spline',
                                     width=1)

# estimate the sampling density
samp_dens = sigpy.gridding(cp.ones_like(data_echo_1.ravel()),
                           k_array.reshape(-1, 3),
                           ishape,
                           kernel='kaiser_bessel')

# correct for sampling density
data_echo_1_gridded_corr = data_echo_1_gridded.copy()
data_echo_1_gridded_corr[samp_dens > 0] /= samp_dens[samp_dens > 0]
data_echo_2_gridded_corr = data_echo_2_gridded.copy()
data_echo_2_gridded_corr[samp_dens > 0] /= samp_dens[samp_dens > 0]

# ifft of sampling density corrected gridded data
ifft_op = sigpy.linop.IFFT(ishape)
ifft_scale = 1.
ifft1 = ifft_scale * ifft_op(data_echo_1_gridded_corr)
ifft2 = ifft_scale * ifft_op(data_echo_2_gridded_corr)

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# setup the operators for reconstruction and regularization

# scale is needed to get normalized operator
recon_operator = (0.00884) * sigpy.linop.NUFFT(
    ishape, k_array.reshape(-1, 3), oversamp=1.25, width=4)

# set up the operator for regularization
G = (1 / np.sqrt(12)) * sigpy.linop.FiniteDifference(ishape, axes=None)
#------------------------------------------------------
# reconstruct the first echo without T2* decay modeling

A = sigpy.linop.Vstack([recon_operator, G])

if regularization_norm_non_anatomical == 'L2':
    proxg = sigpy.prox.L2Reg(G.oshape, lamda=beta_non_anatomical)
elif regularization_norm_non_anatomical == 'L1':
    proxg = sigpy.prox.L1Reg(G.oshape, lamda=beta_non_anatomical)
else:
    raise ValueError('unknown regularization norm')

# estimate norm of the nufft operator if not given
if nufft_scale is None:
    max_eig = sigpy.app.MaxEig(A.H * A,
                               dtype=cp.complex128,
                               device=k_array.device,
                               max_iter=30,
                               show_pbar=True).run()
    nufft_scale = np.sqrt(max_eig)

proxfc1 = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_1.shape, 1, y=-data_echo_1),
    sigpy.prox.Conj(proxg)
])
u1 = cp.zeros(A.oshape, dtype=data_echo_1.dtype)

outfile1 = odir / f'recon_echo_1_no_decay_model_{regularization_norm_non_anatomical}_{beta_non_anatomical:.1E}.npy'

if not outfile1.exists():
    alg1 = sigpy.alg.PrimalDualHybridGradient(proxfc=proxfc1,
                                              proxg=sigpy.prox.NoOp(A.ishape),
                                              A=A,
                                              AH=A.H,
                                              x=deepcopy(ifft1),
                                              u=u1,
                                              tau=1. / sigma,
                                              sigma=sigma,
                                              max_iter=max_num_iter)

    print('recon echo 1 - no T2* modeling')
    for i in range(max_num_iter):
        print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
        alg1.update()
    print('')

    recon_echo_1_wo_decay_model = alg1.x

    # save the recon
    cp.save(outfile1, recon_echo_1_wo_decay_model)
else:
    recon_echo_1_wo_decay_model = cp.load(outfile1)

#-----------------------------------------------------
proxfc2 = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_1.shape, 1, y=-data_echo_2),
    sigpy.prox.Conj(proxg)
])
u2 = cp.zeros(A.oshape, dtype=data_echo_2.dtype)

outfile2 = odir / f'recon_echo_2_no_decay_model_{regularization_norm_non_anatomical}_{beta_non_anatomical:.1E}.npy'

if not outfile2.exists():
    alg2 = sigpy.alg.PrimalDualHybridGradient(proxfc=proxfc2,
                                              proxg=sigpy.prox.NoOp(A.ishape),
                                              A=A,
                                              AH=A.H,
                                              x=deepcopy(ifft2),
                                              u=u2,
                                              tau=1. / sigma,
                                              sigma=sigma,
                                              max_iter=max_num_iter)

    print('recon echo 2 - no T2* modeling')
    for i in range(max_num_iter):
        print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
        alg2.update()
    print('')

    recon_echo_2_wo_decay_model = alg2.x

    cp.save(outfile2, recon_echo_2_wo_decay_model)
else:
    recon_echo_2_wo_decay_model = cp.load(outfile2)
#---------------------------------------------------------------------

ims = dict(vmin=0, vmax=3.5, cmap='Greys_r')
vi = pv.ThreeAxisViewer([
    np.flip(x, (0, 1)) for x in [
        np.abs(cp.asnumpy(recon_echo_1_wo_decay_model)),
        np.abs(cp.asnumpy(recon_echo_2_wo_decay_model))
    ]
],
                        imshow_kwargs=ims)

"""minimal script that shows how to solve L2square data fidelity + anatomical DTV prior"""

import argparse
import h5py
import json
from pathlib import Path
from copy import deepcopy
import numpy as np
import cupy as cp
import sigpy

import pymirc.viewer as pv

from utils import setup_blob_phantom, setup_brainweb_phantom
from utils_sigpy import nufft_t2star_operator

#--------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--regularization_norm',
                    type=str,
                    default='L2',
                    choices=['L1', 'L2'])
parser.add_argument('--betas',
                    type=float,
                    default=[2e-2, 4e-2, 8e-2, 16e-2, 32e-2],
                    nargs='+')
parser.add_argument('--max_num_iter', type=int, default=500)
parser.add_argument('--noise_level', type=float, default=3e-2)
parser.add_argument('--phantom',
                    type=str,
                    default='brainweb',
                    choices=['brainweb', 'blob'])
parser.add_argument('--no_decay', action='store_true')
parser.add_argument('--sigma', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument(
    '--readout_type',
    type=str,
    default='radial7',
    choices=['radial7', 'radial2', 'radial4', 'radial16', 'tpi'])
args = parser.parse_args()

regularization_norm = args.regularization_norm
betas = args.betas
max_num_iter = args.max_num_iter
noise_level = args.noise_level
phantom = args.phantom
no_decay = args.no_decay
sigma = args.sigma
seed = args.seed
readout_type = args.readout_type

cp.random.seed(seed)

#---------------------------------------------------------------
# fixed parameters

simshape = (160, 160, 160)
ishape = (128, 128, 128)

field_of_view_cm: float = 22.

# 10us sampling time
acq_sampling_time_ms: float = 0.01

# echo times in ms
echo_time_1_ms = 0.5
echo_time_2_ms = 5.

# scaling factor for nufft operators such that the norm of the recon
# operator without decay modeling is approx. 1
# extra np.sqrt(2.1) is because radial readout operator has bigger norm
nufft_scale = 0.00884 / np.sqrt(2.1)

# signal fraction decaying with short T2* time
short_fraction = 0.6

time_bin_width_ms: float = 0.25

# read the data root directory from the config file
with open('.simulation_config.json', 'r') as f:
    data_root_dir: str = json.load(f)['data_root_dir']

odir = Path(
    'run_radial') / f'i_{max_num_iter:04}_nl_{noise_level:.1E}_s_{seed:03}'
odir.mkdir(exist_ok=True, parents=True)

with open(odir / 'config.json', 'w') as f:
    json.dump(vars(args), f, indent=4)

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# setup the image

if no_decay:
    decay_suffix = '_no_decay'
    T2long_ms_csf: float = 1e7
    T2long_ms_gm: float = 1e7
    T2long_ms_wm: float = 1e7
    T2short_ms_csf: float = 1e7
    T2short_ms_gm: float = 1e7
    T2short_ms_wm: float = 1e7
else:
    decay_suffix = ''
    T2long_ms_csf: float = 50.
    T2long_ms_gm: float = 15.
    T2long_ms_wm: float = 18.
    T2short_ms_csf: float = 50.
    T2short_ms_gm: float = 8.
    T2short_ms_wm: float = 9.

field_of_view_cm: float = 22.
phantom_data_path: Path = Path(data_root_dir) / 'brainweb54'

# (1) setup the brainweb phantom with the given simulation matrix size
if phantom == 'brainweb':
    x, t1_image, T2short_ms, T2long_ms = setup_brainweb_phantom(
        simshape[0],
        phantom_data_path,
        field_of_view_cm=field_of_view_cm,
        T2long_ms_csf=T2long_ms_csf,
        T2long_ms_gm=T2long_ms_gm,
        T2long_ms_wm=T2long_ms_wm,
        T2short_ms_csf=T2short_ms_csf,
        T2short_ms_gm=T2short_ms_gm,
        T2short_ms_wm=T2short_ms_wm)
elif phantom == 'blob':
    x, t1_image, T2short_ms, T2long_ms = setup_blob_phantom(simshape[0])
else:
    raise ValueError

# move image to GPU
x = cp.asarray(x.astype(np.complex128))

true_ratio_image_short = cp.array(
    np.exp(-(echo_time_2_ms - echo_time_1_ms) / T2short_ms))
true_ratio_image_long = cp.array(
    np.exp(-(echo_time_2_ms - echo_time_1_ms) / T2long_ms))

cp.save(odir / 'na_gt.npy', x)
cp.save(odir / 't1.npy', t1_image)
cp.save(odir / 'true_ratio_short.npy', true_ratio_image_short)
cp.save(odir / 'true_ratio_long.npy', true_ratio_image_long)

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# data simulation block
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# time sampling step in micro seconds
dt_us = 16.
# gamma by 2pi in MHz/T
gamma_by_2pi_MHz_T: float = 11.262

# kmax for 64 matrix (edge)
kmax_1_cm = 1 / (2 * field_of_view_cm / 64)

if readout_type.startswith('radial'):
    # compute radial gradients
    grads_T_m = np.zeros((2272, 1596, 3))

    # radial gradient needed to kmax for 128 matrix
    # uses a gradient of 7 G/cm
    gmax_T_m_rad = float(readout_type.split('radial')[1]) / 10000

    for i in range(grads_T_m.shape[1]):
        phi = np.random.rand() * 2 * np.pi
        cos_theta = np.random.rand() * 2 - 1
        sin_theta = np.sqrt(1 - cos_theta**2)

        grads_T_m[:, i, 0] = gmax_T_m_rad * sin_theta * np.cos(phi)
        grads_T_m[:, i, 1] = gmax_T_m_rad * sin_theta * np.sin(phi)
        grads_T_m[:, i, 2] = gmax_T_m_rad * cos_theta
elif readout_type == 'tpi':
    # read gradients in T/m with shape (num_samples_per_readout, num_readouts, 3)
    with h5py.File(Path(data_root_dir) / 'tpi_gradients/ak_grad56.h5',
                   'r') as data:
        grads_T_m = np.transpose(data['/gradients'][:], (2, 1, 0))
else:
    raise ValueError

# k values in 1/cm with shape (num_samples_per_readout, num_readouts, 3)
k_1_cm = 0.01 * np.cumsum(grads_T_m, axis=0) * dt_us * gamma_by_2pi_MHz_T

k_1_cm_abs = np.linalg.norm(k_1_cm, axis=2)
print(f'readout kmax: {k_1_cm_abs.max():.2f}')
print(f'64 kmax: {kmax_1_cm:.2f}')

##################################################
recon_operator = nufft_t2star_operator(
    ishape,
    k_1_cm,
    field_of_view_cm=field_of_view_cm,
    acq_sampling_time_ms=acq_sampling_time_ms,
    time_bin_width_ms=time_bin_width_ms,
    scale=nufft_scale,
    add_mirrored_coordinates=False,
    echo_time_1_ms=echo_time_1_ms,
    echo_time_2_ms=echo_time_2_ms)
##################################################

# setup the data operators for the 1/2 echo using the short T2* time
data_operator_1_short, data_operator_2_short = nufft_t2star_operator(
    simshape,
    k_1_cm,
    field_of_view_cm=field_of_view_cm,
    acq_sampling_time_ms=acq_sampling_time_ms,
    time_bin_width_ms=time_bin_width_ms,
    scale=nufft_scale,
    add_mirrored_coordinates=False,
    echo_time_1_ms=echo_time_1_ms,
    echo_time_2_ms=echo_time_2_ms,
    ratio_image=true_ratio_image_short)

# setup the data operators for the 1/2 echo using the long T2* time
data_operator_1_long, data_operator_2_long = nufft_t2star_operator(
    simshape,
    k_1_cm,
    field_of_view_cm=field_of_view_cm,
    acq_sampling_time_ms=acq_sampling_time_ms,
    time_bin_width_ms=time_bin_width_ms,
    scale=nufft_scale,
    add_mirrored_coordinates=False,
    echo_time_1_ms=echo_time_1_ms,
    echo_time_2_ms=echo_time_2_ms,
    ratio_image=true_ratio_image_long)

#--------------------------------------------------------------------------
# simulate noise-free data
data_echo_1 = short_fraction * data_operator_1_short(x) + (
    1 - short_fraction) * data_operator_1_long(x)
data_echo_2 = short_fraction * data_operator_2_short(x) + (
    1 - short_fraction) * data_operator_2_long(x)

# scale data to account for difference in simulation and recon matrix sizes
# related to np.fft.fft(norm = 'ortho')
data_echo_1 *= np.sqrt(ishape[0] / simshape[0])**(3)
data_echo_2 *= np.sqrt(ishape[0] / simshape[0])**(3)

# add noise to the data
nl = noise_level * cp.abs(data_echo_1.max())
data_echo_1 += nl * (cp.random.randn(*data_echo_1.shape) +
                     1j * cp.random.randn(*data_echo_1.shape))
data_echo_2 += nl * (cp.random.randn(*data_echo_2.shape) +
                     1j * cp.random.randn(*data_echo_2.shape))

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# recon block
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# grid data for conventional IFFT recon
data_echo_1_gridded = sigpy.gridding(data_echo_1,
                                     cp.asarray(k_1_cm.reshape(-1, 3)) *
                                     field_of_view_cm,
                                     ishape,
                                     kernel='spline',
                                     width=1)
data_echo_2_gridded = sigpy.gridding(data_echo_2,
                                     cp.asarray(k_1_cm.reshape(-1, 3)) *
                                     field_of_view_cm,
                                     ishape,
                                     kernel='spline',
                                     width=1)
samp_dens = sigpy.gridding(cp.ones_like(data_echo_1),
                           cp.asarray(k_1_cm.reshape(-1, 3)) *
                           field_of_view_cm,
                           ishape,
                           kernel='spline')
ifft_op = sigpy.linop.IFFT(ishape)

data_echo_1_gridded_corr = data_echo_1_gridded.copy()
data_echo_1_gridded_corr[samp_dens > 0] /= samp_dens[samp_dens > 0]
data_echo_2_gridded_corr = data_echo_2_gridded.copy()
data_echo_2_gridded_corr[samp_dens > 0] /= samp_dens[samp_dens > 0]

# perform IFFT recon (without correction for fall-of due to k-space interpolation)
ifft_scale = 909.1 / 2
ifft1 = ifft_scale * ifft_op(data_echo_1_gridded_corr)
ifft2 = ifft_scale * ifft_op(data_echo_2_gridded_corr)

# calculated filtered IFFT

del data_operator_1_short
del data_operator_1_long
del data_operator_2_short
del data_operator_2_long

cp.save(odir / 'ifft1.npy', ifft1)
cp.save(odir / 'ifft2.npy', ifft2)

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# setup projected gradient operator for DTV

# set up the operator for regularization
G = (1 / np.sqrt(12)) * sigpy.linop.FiniteDifference(ishape, axes=None)

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# recons without decay model and non-anatomical prior
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# run iterative recon of first echo with non-anatomical prior
A = sigpy.linop.Vstack([recon_operator, G])

# estimate the max eigenvalue of A.H * A for the step sizes of PDHG
max_eig = sigpy.app.MaxEig(A.H * A,
                           dtype=cp.complex128,
                           device=x.device,
                           max_iter=30).run()

recons = cp.zeros((len(betas), ) + ishape, dtype=cp.complex128)

for ib, beta in enumerate(betas):
    outfile1 = odir / f'recon_echo_1_no_decay_model_{readout_type}_{regularization_norm}_{beta:.1E}_{max_num_iter}.npz'

    if regularization_norm == 'L2':
        prox_reg_non_anatomical = sigpy.prox.L2Reg(G.oshape, lamda=beta)
    elif regularization_norm == 'L1':
        prox_reg_non_anatomical = sigpy.prox.L1Reg(G.oshape, lamda=beta)
    else:
        raise ValueError('unknown regularization norm')

    proxfc1 = sigpy.prox.Stack([
        sigpy.prox.L2Reg(data_echo_1.shape, 1, y=-data_echo_1),
        sigpy.prox.Conj(prox_reg_non_anatomical)
    ])

    u1 = cp.zeros(A.oshape, dtype=data_echo_1.dtype)

    if not outfile1.exists():
        alg1 = sigpy.alg.PrimalDualHybridGradient(proxfc=proxfc1,
                                                  proxg=sigpy.prox.NoOp(
                                                      A.ishape),
                                                  A=A,
                                                  AH=A.H,
                                                  x=deepcopy(ifft1),
                                                  u=u1,
                                                  tau=1. / (sigma * max_eig),
                                                  sigma=sigma,
                                                  max_iter=max_num_iter)

        print('recon echo 1 - no T2* modeling')
        for i in range(max_num_iter):
            print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
            alg1.update()
        print('')

        cp.savez(outfile1, x=alg1.x, u=u1)
        recon_echo_1_wo_decay_model = alg1.x
    else:
        d1 = cp.load(outfile1)
        recon_echo_1_wo_decay_model = d1['x']
        u1 = d1['u']

    recons[ib, ...] = recon_echo_1_wo_decay_model

vi = pv.ThreeAxisViewer(np.abs(cp.asnumpy(recons)),
                        imshow_kwargs=dict(vmin=0, vmax=3.5, cmap='Greys_r'))

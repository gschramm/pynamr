"""minimal script that shows how to solve L2square data fidelity + anatomical DTV prior"""

import argparse
import h5py
import json
from pathlib import Path
from copy import deepcopy
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndimage
import sigpy

import pymirc.viewer as pv
from pymirc.image_operations import zoom3d

from utils import setup_blob_phantom, setup_brainweb_phantom, hann
from utils_sigpy import nufft_t2star_operator

#--------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--regularization_operator',
                    type=str,
                    default='projected_gradient',
                    choices=['projected_gradient', 'gradient'])
parser.add_argument('--regularization_norm',
                    type=str,
                    default='L1',
                    choices=['L1', 'L2'])
parser.add_argument('--beta', type=float, default=2e-2)
parser.add_argument('--max_num_iter', type=int, default=300)
parser.add_argument('--noise_level', type=float, default=3e-2)
parser.add_argument('--phantom',
                    type=str,
                    default='brainweb',
                    choices=['brainweb', 'blob'])
parser.add_argument('--no_decay', action='store_true')
parser.add_argument('--sigma', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

regularization_operator = args.regularization_operator
regularization_norm = args.regularization_norm
beta = args.beta
max_num_iter = args.max_num_iter
noise_level = args.noise_level
phantom = args.phantom
no_decay = args.no_decay
sigma = args.sigma
seed = args.seed

add_mismatches = True

regularization_norm_non_anatomical = 'L2'
beta_non_anatomical = 2e-1

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
nufft_scale = 0.00884

# signal fraction decaying with short T2* time
short_fraction = 0.6

time_bin_width_ms: float = 0.25

# read the data root directory from the config file
with open('.simulation_config.json', 'r') as f:
    data_root_dir: str = json.load(f)['data_root_dir']

odir = Path('run') / f'i_{max_num_iter:04}_nl_{noise_level:.1E}'
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

if add_mismatches:
    # add mismatching structures
    tmp = np.linspace(0, 1, x.shape[0])
    X, Y, Z = np.meshgrid(tmp, tmp, tmp)
    R = np.sqrt((X - 0.681)**2 + (Y - 0.612)**2 + (Z - 0.5)**2)
    x[R < 0.02] = 1.5

    R2 = np.sqrt((X - 0.7)**2 + (Y - 0.394)**2 + (Z - 0.5)**2)
    t1_image[R2 < 0.02] = 0

    # multiply the T2* times with a correction factor that varies across the FH direction
    tmp = np.linspace(-1, 1, simshape[0])
    X, Y, Z = np.meshgrid(tmp, tmp, tmp)
    corr_field = (2 / np.pi) * np.arctan(20 * (Z + 0.3)) / 2.5 + (1.5 / 2.5)

    T2short_ms *= corr_field
    T2long_ms *= corr_field

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

kmax_1_cm = 1 / (2 * field_of_view_cm / 64)

# read gradients in T/m with shape (num_samples_per_readout, num_readouts, 3)
with h5py.File(Path(data_root_dir) / 'tpi_gradients/ak_grad56.h5',
               'r') as data:
    grads_T_m = np.transpose(data['/gradients'][:], (2, 1, 0))

# time sampling step in micro seconds
dt_us = 16.

# gamma by 2pi in MHz/T
gamma_by_2pi_MHz_T: float = 11.262

# k values in 1/cm with shape (num_samples_per_readout, num_readouts, 3)
k_1_cm = 0.01 * np.cumsum(grads_T_m, axis=0) * dt_us * gamma_by_2pi_MHz_T

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
k_fft_1_cm = np.fft.fftfreq(ishape[0], d=field_of_view_cm / ishape[0])
K0, K1, K2 = np.meshgrid(k_fft_1_cm, k_fft_1_cm, k_fft_1_cm, indexing='ij')
Kabs_fft_1_cm = np.sqrt(K0**2 + K1**2 + K2**2)
filt = cp.asarray(hann(Kabs_fft_1_cm, kmax_1_cm))
ifft1_filt = ifft_scale * ifft_op(data_echo_1_gridded_corr * filt)
ifft2_filt = ifft_scale * ifft_op(data_echo_2_gridded_corr * filt)

#vi = pv.ThreeAxisViewer([np.abs(cp.asnumpy(ifft1)), np.abs(cp.asnumpy(ifft2))])

del data_operator_1_short
del data_operator_1_long
del data_operator_2_short
del data_operator_2_long

cp.save(odir / 'ifft1.npy', ifft1)
cp.save(odir / 'ifft2.npy', ifft2)
cp.save(odir / 'ifft1_filt.npy', ifft1_filt)
cp.save(odir / 'ifft2_filt.npy', ifft2_filt)

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
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

# setup projected gradient operator for DTV

# set up the operator for regularization
G = (1 / np.sqrt(12)) * sigpy.linop.FiniteDifference(ishape, axes=None)

# setup a joint gradient field
prior_image = cp.asarray(zoom3d(t1_image, ishape[0] / simshape[0]))
xi = G(prior_image)

# normalize the real and imaginary part of the joint gradient field
real_norm = cp.linalg.norm(xi.real, axis=0)
imag_norm = cp.linalg.norm(xi.imag, axis=0)

ir = cp.where(real_norm > 0)
ii = cp.where(imag_norm > 0)

for i in range(xi.shape[0]):
    xi[i, ...].real[ir] /= real_norm[ir]
    xi[i, ...].imag[ii] /= imag_norm[ii]

M = sigpy.linop.Multiply(G.oshape, xi)
S = sigpy.linop.Sum(M.oshape, (0, ))
I = sigpy.linop.Identity(M.oshape)

# projection operator
P = I - (M.H * S.H * S * M)

# projected gradient operator
PG = P * G

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# recons without decay model and non-anatomical prior
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# run iterative recon of first echo with non-anatomical prior
A = sigpy.linop.Vstack([recon_operator, G])

if regularization_norm_non_anatomical == 'L2':
    prox_reg_non_anatomical = sigpy.prox.L2Reg(G.oshape,
                                               lamda=beta_non_anatomical)
elif regularization_norm_non_anatomical == 'L1':
    prox_reg_non_anatomical = sigpy.prox.L1Reg(G.oshape,
                                               lamda=beta_non_anatomical)
else:
    raise ValueError('unknown regularization norm')

# reconstruct 1st echo
proxfc1 = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_1.shape, 1, y=-data_echo_1),
    sigpy.prox.Conj(prox_reg_non_anatomical)
])
u1 = cp.zeros(A.oshape, dtype=data_echo_1.dtype)

outfile1 = odir / f'recon_echo_1_no_decay_model_{regularization_norm_non_anatomical}_{beta_non_anatomical:.1E}.npz'

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

    cp.savez(outfile1, x=alg1.x, u=u1)
    recon_echo_1_wo_decay_model = alg1.x
else:
    d1 = cp.load(outfile1)
    recon_echo_1_wo_decay_model = d1['x']
    u1 = d1['u']

# reconstruct 2nd echo
proxfc2 = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_2.shape, 1, y=-data_echo_2),
    sigpy.prox.Conj(prox_reg_non_anatomical)
])
u2 = cp.zeros(A.oshape, dtype=data_echo_2.dtype)

outfile2 = odir / f'recon_echo_2_no_decay_model_{regularization_norm_non_anatomical}_{beta_non_anatomical:.1E}.npz'

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

    cp.savez(outfile2, x=alg2.x, u=u2)
    recon_echo_2_wo_decay_model = alg2.x
else:
    d2 = cp.load(outfile2)
    recon_echo_2_wo_decay_model = d2['x']
    u2 = d2['u']

del A

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# recons without decay model and anatomical prior
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

A = sigpy.linop.Vstack([recon_operator, PG])

if regularization_norm == 'L2':
    prox_reg = sigpy.prox.L2Reg(PG.oshape, lamda=beta)
elif regularization_norm == 'L1':
    prox_reg = sigpy.prox.L1Reg(PG.oshape, lamda=beta)
else:
    raise ValueError('unknown regularization norm')

# recon of first echo
proxfc1 = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_1.shape, 1, y=-data_echo_1),
    sigpy.prox.Conj(prox_reg)
])

u3 = deepcopy(u1)

# reconstruct 1st echo
outfile3 = odir / f'agr_echo_1_no_decay_model_{regularization_norm}_{beta:.1E}.npz'

if not outfile3.exists():
    alg3 = sigpy.alg.PrimalDualHybridGradient(
        proxfc=proxfc1,
        proxg=sigpy.prox.NoOp(A.ishape),
        A=A,
        AH=A.H,
        x=deepcopy(recon_echo_1_wo_decay_model),
        u=u3,
        tau=1. / sigma,
        sigma=sigma,
        max_iter=max_num_iter)

    print('AGR echo 1 - no T2* modeling')
    for i in range(max_num_iter):
        print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
        alg3.update()
    print('')

    cp.savez(outfile3, x=alg3.x, u=u3)
    agr_echo_1_wo_decay_model = alg3.x
else:
    d3 = cp.load(outfile3)
    agr_echo_1_wo_decay_model = d3['x']
    u3 = d3['u']

# recon of second echo
proxfc2 = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_2.shape, 1, y=-data_echo_2),
    sigpy.prox.Conj(prox_reg)
])

u4 = deepcopy(u2)

# reconstruct 2nd echo
outfile4 = odir / f'agr_echo_2_no_decay_model_{regularization_norm}_{beta:.1E}.npz'

if not outfile4.exists():
    alg4 = sigpy.alg.PrimalDualHybridGradient(
        proxfc=proxfc2,
        proxg=sigpy.prox.NoOp(A.ishape),
        A=A,
        AH=A.H,
        x=deepcopy(recon_echo_2_wo_decay_model),
        u=u4,
        tau=1. / sigma,
        sigma=sigma,
        max_iter=max_num_iter)

    print('AGR echo 2 - no T2* modeling')
    for i in range(max_num_iter):
        print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
        alg4.update()
    print('')

    cp.savez(outfile4, x=alg4.x, u=u4)
    agr_echo_2_wo_decay_model = alg4.x
else:
    d4 = cp.load(outfile4)
    agr_echo_2_wo_decay_model = d4['x']
    u4 = d4['u']

del A
del recon_operator

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# recons "true" decay model and anatomical prior
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# approximate "true" monoexp. ratio
true_ratio = short_fraction * true_ratio_image_short + (
    1 - short_fraction) * true_ratio_image_long

# extrapolate true_ratio to recon shape
true_ratio = cp.asarray(
    zoom3d(cp.asnumpy(true_ratio), ishape[0] / true_ratio.shape[0]))

recon_operator_1t, recon_operator_2t = nufft_t2star_operator(
    ishape,
    k_1_cm,
    field_of_view_cm=field_of_view_cm,
    acq_sampling_time_ms=acq_sampling_time_ms,
    time_bin_width_ms=time_bin_width_ms,
    scale=nufft_scale,
    add_mirrored_coordinates=False,
    echo_time_1_ms=echo_time_1_ms,
    echo_time_2_ms=echo_time_2_ms,
    ratio_image=true_ratio)

A = sigpy.linop.Vstack([recon_operator_1t, PG])

# recon of first echo
u5 = deepcopy(u3)

# reconstruct 1st echo
outfile5 = odir / f'agr_echo_1_true_decay_model_{regularization_norm}_{beta:.1E}.npz'

if not outfile5.exists():
    alg5 = sigpy.alg.PrimalDualHybridGradient(
        proxfc=proxfc1,
        proxg=sigpy.prox.NoOp(A.ishape),
        A=A,
        AH=A.H,
        x=deepcopy(agr_echo_1_wo_decay_model),
        u=u5,
        tau=1. / sigma,
        sigma=sigma,
        max_iter=max_num_iter)

    print('AGR echo 1 - "true" T2* modeling')
    for i in range(max_num_iter):
        print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
        alg5.update()
    print('')

    cp.savez(outfile5, x=alg5.x, u=u5)
    agr_echo_1_true_decay_model = alg5.x
else:
    d5 = cp.load(outfile5)
    agr_echo_1_true_decay_model = d5['x']
    u5 = d5['u']

# recon of 2nd echo
u6 = deepcopy(u4)
outfile6 = odir / f'agr_echo_2_true_decay_model_{regularization_norm}_{beta:.1E}.npz'

if not outfile6.exists():
    alg6 = sigpy.alg.PrimalDualHybridGradient(
        proxfc=proxfc2,
        proxg=sigpy.prox.NoOp(A.ishape),
        A=A,
        AH=A.H,
        x=deepcopy(agr_echo_2_wo_decay_model),
        u=u6,
        tau=1. / sigma,
        sigma=sigma,
        max_iter=max_num_iter)

    print('AGR echo 2 - "true" T2* modeling')
    for i in range(max_num_iter):
        print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
        alg6.update()
    print('')

    cp.savez(outfile6, x=alg6.x, u=u6)
    agr_echo_2_true_decay_model = alg6.x
else:
    d6 = cp.load(outfile6)
    agr_echo_2_true_decay_model = d6['x']
    u6 = d6['u']

#-------------------------------------------------------------------------
# calculate the ratio between the two recons without T2* decay modeling
# to estimate a monoexponential T2*

est_ratio = cp.clip(
    cp.abs(agr_echo_2_wo_decay_model) / cp.abs(agr_echo_1_wo_decay_model), 0,
    1)
# set ratio to one in voxels where there is low signal in the first echo
mask = 1 - (cp.abs(agr_echo_1_wo_decay_model) <
            0.05 * cp.abs(agr_echo_1_wo_decay_model).max())

label, num_label = ndimage.label(mask == 1)
size = np.bincount(label.ravel())
biggest_label = size[1:].argmax() + 1
clump_mask = (label == biggest_label)

est_ratio[clump_mask == 0] = 1

cp.save(odir / 'est_ratio.npy', est_ratio)

del A
del recon_operator_1t
del recon_operator_2t

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# recons with "estimated" decay model and anatomical prior
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

recon_operator_1e, recon_operator_2e = nufft_t2star_operator(
    ishape,
    k_1_cm,
    field_of_view_cm=field_of_view_cm,
    acq_sampling_time_ms=acq_sampling_time_ms,
    time_bin_width_ms=time_bin_width_ms,
    scale=nufft_scale,
    add_mirrored_coordinates=False,
    echo_time_1_ms=echo_time_1_ms,
    echo_time_2_ms=echo_time_2_ms,
    ratio_image=est_ratio)

A = sigpy.linop.Vstack([recon_operator_1e, PG])

# recon of first echo
u7 = deepcopy(u5)
outfile7 = odir / f'agr_echo_1_est_decay_model_{regularization_norm}_{beta:.1E}.npz'

if not outfile7.exists():
    alg7 = sigpy.alg.PrimalDualHybridGradient(
        proxfc=proxfc1,
        proxg=sigpy.prox.NoOp(A.ishape),
        A=A,
        AH=A.H,
        x=deepcopy(agr_echo_1_wo_decay_model),
        u=u7,
        tau=1. / sigma,
        sigma=sigma,
        max_iter=max_num_iter)

    print('AGR echo 1 - "estimated" T2* modeling')
    for i in range(max_num_iter):
        print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
        alg7.update()
    print('')

    cp.savez(outfile7, x=alg7.x, u=u7)
    agr_echo_1_est_decay_model = alg7.x
else:
    d7 = cp.load(outfile7)
    agr_echo_1_est_decay_model = d7['x']
    u7 = d7['u']

# recon of second echo
u8 = deepcopy(u6)
outfile8 = odir / f'agr_echo_2_est_decay_model_{regularization_norm}_{beta:.1E}.npz'

if not outfile8.exists():
    alg8 = sigpy.alg.PrimalDualHybridGradient(
        proxfc=proxfc2,
        proxg=sigpy.prox.NoOp(A.ishape),
        A=A,
        AH=A.H,
        x=deepcopy(agr_echo_2_wo_decay_model),
        u=u8,
        tau=1. / sigma,
        sigma=sigma,
        max_iter=max_num_iter)

    print('AGR echo 2 - "estimated" T2* modeling')
    for i in range(max_num_iter):
        print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
        alg8.update()
    print('')

    cp.savez(outfile8, x=alg8.x, u=u8)
    agr_echo_2_est_decay_model = alg8.x
else:
    d8 = cp.load(outfile8)
    agr_echo_2_est_decay_model = d8['x']
    u8 = d8['u']

#-----------------------------------------------------------------------------

ims = 3 * [dict(vmin=0, vmax=3.5, cmap='Greys_r')]

vi1 = pv.ThreeAxisViewer([
    np.flip(x, (0, 1)) for x in [
        np.abs(cp.asnumpy(ifft1)),
        np.abs(cp.asnumpy(ifft1_filt)),
        np.abs(cp.asnumpy(recon_echo_1_wo_decay_model)),
    ]
],
                         imshow_kwargs=ims)

vi2 = pv.ThreeAxisViewer([
    np.flip(x, (0, 1)) for x in [
        np.abs(cp.asnumpy(agr_echo_1_wo_decay_model)),
        np.abs(cp.asnumpy(agr_echo_1_true_decay_model)),
        np.abs(cp.asnumpy(agr_echo_1_est_decay_model)),
    ]
],
                         imshow_kwargs=ims)

vi3 = pv.ThreeAxisViewer([
    np.flip(x, (0, 1)) for x in [
        np.abs(cp.asnumpy(ifft2)),
        np.abs(cp.asnumpy(ifft2_filt)),
        np.abs(cp.asnumpy(recon_echo_2_wo_decay_model)),
    ]
],
                         imshow_kwargs=ims)

vi4 = pv.ThreeAxisViewer([
    np.flip(x, (0, 1)) for x in [
        np.abs(cp.asnumpy(agr_echo_2_wo_decay_model)),
        np.abs(cp.asnumpy(agr_echo_2_true_decay_model)),
        np.abs(cp.asnumpy(agr_echo_2_est_decay_model)),
    ]
],
                         imshow_kwargs=ims)

vi5 = pv.ThreeAxisViewer([
    np.flip(x, (0, 1))
    for x in [cp.asnumpy(true_ratio),
              cp.asnumpy(est_ratio)]
],
                         imshow_kwargs=dict(vmin=0, vmax=1, cmap='Greys_r'))

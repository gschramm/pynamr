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
from pymirc.image_operations import zoom3d

import pymirc.viewer as pv
import matplotlib.pyplot as plt

from utils import setup_blob_phantom, setup_brainweb_phantom, kb_rolloff
from utils_sigpy import nufft_t2star_operator

from scipy.ndimage import binary_erosion
#--------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--max_num_iter', type=int, default=200)
parser.add_argument('--noise_level', type=float, default=1e-2)
parser.add_argument('--phantom',
                    type=str,
                    default='brainweb',
                    choices=['brainweb', 'blob'])
parser.add_argument('--no_decay', action='store_true')
parser.add_argument('--sigma', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--beta_anatomical', type=float, default=1e-2)
parser.add_argument('--beta_non_anatomical', type=float, default=3e-1)
args = parser.parse_args()

max_num_iter = args.max_num_iter
noise_level = args.noise_level
phantom = args.phantom
no_decay = args.no_decay
sigma = args.sigma
seed = args.seed
beta_anatomical = args.beta_anatomical
beta_non_anatomical = args.beta_non_anatomical

cp.random.seed(seed)

#---------------------------------------------------------------
# fixed parameters

sim_shape = (160, 160, 160)
iter_shape = (128, 128, 128)
grid_shape = (64, 64, 64)

field_of_view_cm: float = 22.

# 10us sampling time
acq_sampling_time_ms: float = 0.016

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

odir = Path(
    'run_brainweb'
) / f'{phantom}_nodecay_{no_decay}_i_{max_num_iter:04}_nl_{noise_level:.1E}_s_{seed:03}'
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
        sim_shape[0],
        phantom_data_path,
        field_of_view_cm=field_of_view_cm,
        T2long_ms_csf=T2long_ms_csf,
        T2long_ms_gm=T2long_ms_gm,
        T2long_ms_wm=T2long_ms_wm,
        T2short_ms_csf=T2short_ms_csf,
        T2short_ms_gm=T2short_ms_gm,
        T2short_ms_wm=T2short_ms_wm)
elif phantom == 'blob':
    x, t1_image, T2short_ms, T2long_ms = setup_blob_phantom(sim_shape[0],
                                                            radius=0.65)
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

# read gradients in T/m with shape (num_samples_per_readout, num_readouts, 3)
with h5py.File(Path(data_root_dir) / 'tpi_gradients/ak_grad56.h5',
               'r') as data:
    grads_T_m = np.transpose(data['/gradients'][:], (2, 1, 0))

# k values in 1/cm with shape (num_samples_per_readout, num_readouts, 3)
k_1_cm = 0.01 * np.cumsum(grads_T_m, axis=0) * dt_us * gamma_by_2pi_MHz_T

k_1_cm_abs = np.linalg.norm(k_1_cm, axis=2)
print(f'readout kmax: {k_1_cm_abs.max():.2f}')
print(f'64 kmax: {kmax_1_cm:.2f}')

# setup the data operators for the 1/2 echo using the short T2* time
data_operator_1_short, data_operator_2_short = nufft_t2star_operator(
    sim_shape,
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
    sim_shape,
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
data_echo_1 *= np.sqrt(iter_shape[0] / sim_shape[0])**(3)
data_echo_2 *= np.sqrt(iter_shape[0] / sim_shape[0])**(3)

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
kernel = 'kaiser_bessel'
width = 2
param = 9.14

data_echo_1_gridded = sigpy.gridding(data_echo_1,
                                     cp.asarray(k_1_cm.reshape(-1, 3)) *
                                     field_of_view_cm,
                                     grid_shape,
                                     kernel=kernel,
                                     width=width,
                                     param=param)
data_echo_2_gridded = sigpy.gridding(data_echo_2,
                                     cp.asarray(k_1_cm.reshape(-1, 3)) *
                                     field_of_view_cm,
                                     grid_shape,
                                     kernel=kernel,
                                     width=width,
                                     param=param)
samp_dens = sigpy.gridding(cp.ones_like(data_echo_1),
                           cp.asarray(k_1_cm.reshape(-1, 3)) *
                           field_of_view_cm,
                           grid_shape,
                           kernel=kernel,
                           width=width,
                           param=param)
ifft_op = sigpy.linop.IFFT(grid_shape)

data_echo_1_gridded_corr = data_echo_1_gridded.copy()
data_echo_1_gridded_corr[samp_dens > 0] /= samp_dens[samp_dens > 0]
data_echo_2_gridded_corr = data_echo_2_gridded.copy()
data_echo_2_gridded_corr[samp_dens > 0] /= samp_dens[samp_dens > 0]

# perform IFFT recon (without correction for fall-of due to k-space interpolation)
if phantom == 'brainweb':
    ifft_scale = 62.5 / np.sqrt(8)
elif phantom == 'blob':
    ifft_scale = 80.5 / np.sqrt(8)
else:
    raise ValueError

ifft1 = ifft_scale * ifft_op(data_echo_1_gridded_corr)
ifft2 = ifft_scale * ifft_op(data_echo_2_gridded_corr)

tmp_x = cp.linspace(-width / 2, width / 2, grid_shape[0])
TMP_X, TMP_Y, TMP_Z = cp.meshgrid(tmp_x, tmp_x, tmp_x)
R = cp.sqrt(TMP_X**2 + TMP_Y**2 + TMP_Z**2)
R = cp.clip(R, 0, tmp_x.max())
#TODO: understand why factor 1.6 is needed when regridding in 3D
interpolation_correction_field = kb_rolloff(1.6 * R, param)
interpolation_correction_field /= interpolation_correction_field.max()

## check roll-off correction (e.g. in blob phantom)
#r = cp.asnumpy(R).ravel()
#c = cp.asnumpy(interpolation_correction_field).ravel()
#p = cp.asnumpy(cp.abs(ifft1)).ravel()
#p /= p.max()
#
#import matplotlib.pyplot as plt
#
#fig, ax = plt.subplots()
#ax.plot(r, c, '.')
#ax.plot(r, p, '.')
#fig.show()

ifft1 /= interpolation_correction_field
ifft2 /= interpolation_correction_field

del data_operator_1_short
del data_operator_1_long
del data_operator_2_short
del data_operator_2_long
cp.save(odir / f'ifft1.npy', ifft1)
cp.save(odir / f'ifft2.npy', ifft2)

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# recon of first echo without decay model and non-anatomical prior
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

nufft_single_echo_no_decay = nufft_t2star_operator(
    iter_shape,
    k_1_cm,
    field_of_view_cm=field_of_view_cm,
    acq_sampling_time_ms=acq_sampling_time_ms,
    time_bin_width_ms=time_bin_width_ms,
    scale=nufft_scale,
    add_mirrored_coordinates=False,
    echo_time_1_ms=echo_time_1_ms,
    echo_time_2_ms=echo_time_2_ms)

# setup 3D finite difference operator with norm 1
G = (1 / np.sqrt(12)) * sigpy.linop.FiniteDifference(iter_shape, axes=None)
prox_reg_non_anatomical = sigpy.prox.L2Reg(G.oshape, lamda=beta_non_anatomical)

# setup complete forward model and proximal operator
A = sigpy.linop.Vstack([nufft_single_echo_no_decay, G])

proxfc1 = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_1.shape, 1, y=-data_echo_1),
    sigpy.prox.Conj(prox_reg_non_anatomical)
])

# setup the dual variable
u_e1_no_decay = cp.zeros(A.oshape, dtype=data_echo_1.dtype)

ofile_e1_no_decay = odir / f'recon_echo_1_no_decay_model_{max_num_iter}_{beta_non_anatomical}.npz'

if not ofile_e1_no_decay.exists():
    alg_e1_no_decay = sigpy.alg.PrimalDualHybridGradient(
        proxfc=proxfc1,
        proxg=sigpy.prox.NoOp(A.ishape),
        A=A,
        AH=A.H,
        x=cp.zeros(A.ishape, dtype=cp.complex128),
        u=u_e1_no_decay,
        tau=1. / sigma,
        sigma=sigma,
        max_iter=max_num_iter)

    print('recon echo 1 - no T2* modeling')
    for i in range(max_num_iter):
        print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
        alg_e1_no_decay.update()
    print('')

    cp.savez(ofile_e1_no_decay, x=alg_e1_no_decay.x, u=u_e1_no_decay)
    r_e1_no_decay = alg_e1_no_decay.x
else:
    d = cp.load(ofile_e1_no_decay)
    r_e1_no_decay = d['x']
    u_e1_no_decay = d['u']

del A

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# recon of first echo without decay model and anatomical prior
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

#----------------------------------
# setup projected gradient operator
#----------------------------------

prior_image = cp.asarray(zoom3d(t1_image, iter_shape[0] / sim_shape[0]))
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

prox_reg_anatomical = sigpy.prox.L1Reg(PG.oshape, lamda=beta_anatomical)

# setup complete forward model and proximal operator
A = sigpy.linop.Vstack([nufft_single_echo_no_decay, PG])

proxfc1 = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_1.shape, 1, y=-data_echo_1),
    sigpy.prox.Conj(prox_reg_anatomical)
])

# setup the dual variable
u_e1_no_decay_agr = deepcopy(u_e1_no_decay)

ofile_e1_no_decay_agr = odir / f'agr_echo_1_no_decay_model_{max_num_iter}_{beta_anatomical}.npz'

if not ofile_e1_no_decay_agr.exists():
    alg_e1_no_decay_agr = sigpy.alg.PrimalDualHybridGradient(
        proxfc=proxfc1,
        proxg=sigpy.prox.NoOp(A.ishape),
        A=A,
        AH=A.H,
        x=deepcopy(r_e1_no_decay),
        u=u_e1_no_decay_agr,
        tau=1. / sigma,
        sigma=sigma,
        max_iter=max_num_iter)

    print('agr echo 1 - no T2* modeling')
    for i in range(max_num_iter):
        print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
        alg_e1_no_decay_agr.update()
    print('')

    cp.savez(ofile_e1_no_decay_agr,
             x=alg_e1_no_decay_agr.x,
             u=u_e1_no_decay_agr)
    agr_e1_no_decay = alg_e1_no_decay_agr.x
else:
    d = cp.load(ofile_e1_no_decay_agr)
    agr_e1_no_decay = d['x']
    u_e1_no_decay_agr = d['u']

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# recon of second echo without decay model and anatomical prior
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

proxfc2 = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_2.shape, 1, y=-data_echo_2),
    sigpy.prox.Conj(prox_reg_anatomical)
])

# setup the dual variable
u_e2_no_decay_agr = deepcopy(u_e1_no_decay_agr)

ofile_e2_no_decay_agr = odir / f'agr_echo_2_no_decay_model_{max_num_iter}_{beta_anatomical}.npz'

if not ofile_e2_no_decay_agr.exists():
    alg_e2_no_decay_agr = sigpy.alg.PrimalDualHybridGradient(
        proxfc=proxfc2,
        proxg=sigpy.prox.NoOp(A.ishape),
        A=A,
        AH=A.H,
        x=deepcopy(agr_e1_no_decay),
        u=u_e2_no_decay_agr,
        tau=1. / sigma,
        sigma=sigma,
        max_iter=max_num_iter)

    print('agr echo 2 - no T2* modeling')
    for i in range(max_num_iter):
        print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
        alg_e2_no_decay_agr.update()
    print('')

    cp.savez(ofile_e2_no_decay_agr,
             x=alg_e2_no_decay_agr.x,
             u=u_e2_no_decay_agr)
    agr_e2_no_decay = alg_e2_no_decay_agr.x
else:
    d = cp.load(ofile_e2_no_decay_agr)
    agr_e2_no_decay = d['x']
    u_e2_no_decay_agr = d['u']

del A
del nufft_single_echo_no_decay

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# estimate the ratio between first and second echo (AGR)
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

est_ratio = cp.clip(cp.abs(agr_e2_no_decay) / cp.abs(agr_e1_no_decay), 0, 1)
# set ratio to one in voxels where there is low signal in the first echo
mask = 1 - (cp.abs(agr_e1_no_decay) < 0.05 * cp.abs(agr_e1_no_decay).max())

label, num_label = ndimage.label(mask == 1)
size = np.bincount(label.ravel())
biggest_label = size[1:].argmax() + 1
clump_mask = (label == biggest_label)

est_ratio[clump_mask == 0] = 1

cp.save(odir / f'est_ratio_{max_num_iter}_{beta_anatomical}.npy', est_ratio)

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# AGR of both echos including T2* model
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

nufft_echo_1, nufft_echo_2 = nufft_t2star_operator(
    iter_shape,
    k_1_cm,
    field_of_view_cm=field_of_view_cm,
    acq_sampling_time_ms=acq_sampling_time_ms,
    time_bin_width_ms=time_bin_width_ms,
    scale=nufft_scale,
    add_mirrored_coordinates=False,
    echo_time_1_ms=echo_time_1_ms,
    echo_time_2_ms=echo_time_2_ms,
    ratio_image=est_ratio)

# the two echos usually have different phases
# we correct for this by multiplying by the negative estimated phases
phase_fac_1 = cp.exp(1j * cp.angle(agr_e1_no_decay))
phase_fac_2 = cp.exp(1j * cp.angle(agr_e2_no_decay))

A = sigpy.linop.Vstack([
    nufft_echo_1 * sigpy.linop.Multiply(iter_shape, phase_fac_1),
    nufft_echo_2 * sigpy.linop.Multiply(iter_shape, phase_fac_2), PG
])

proxfcb = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_1.shape, 1, y=-data_echo_1),
    sigpy.prox.L2Reg(data_echo_2.shape, 1, y=-data_echo_2),
    sigpy.prox.Conj(prox_reg_anatomical)
])

ub = cp.concatenate((u_e1_no_decay_agr[:data_echo_1.size],
                     u_e2_no_decay_agr[:data_echo_2.size],
                     u_e1_no_decay_agr[data_echo_1.size:]))

ofile_agr_both_echos = odir / f'agr_both_echos_w_decay_model_{beta_anatomical:.1E}.npz'

if not ofile_agr_both_echos.exists():
    algb = sigpy.alg.PrimalDualHybridGradient(proxfc=proxfcb,
                                              proxg=sigpy.prox.NoOp(A.ishape),
                                              A=A,
                                              AH=A.H,
                                              x=deepcopy(agr_e1_no_decay),
                                              u=ub,
                                              tau=0.7 / sigma,
                                              sigma=sigma,
                                              max_iter=max_num_iter)

    print('AGR both echos - "estimated" T2* modeling')
    for i in range(max_num_iter):
        print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
        algb.update()
    print('')

    cp.savez(ofile_agr_both_echos, x=algb.x, u=ub)
    agr_both_echos_w_decay_model = algb.x
else:
    d = cp.load(ofile_agr_both_echos)
    agr_both_echos_w_decay_model = d['x']
    u = d['u']

del A
del nufft_echo_1
del nufft_echo_2

#-------------------------------------------------------------------------------
# show results
from scipy.ndimage import zoom

a = zoom(cp.asnumpy(cp.flip(cp.abs(ifft1), (0, 1))), 256 / grid_shape[0])
a2 = zoom(cp.asnumpy(cp.flip(cp.abs(r_e1_no_decay), (0, 1))),
          256 / iter_shape[0])
b = zoom(cp.asnumpy(cp.flip(cp.abs(agr_e1_no_decay), (0, 1))),
         256 / iter_shape[0])
b3 = zoom(cp.asnumpy(cp.flip(cp.abs(agr_both_echos_w_decay_model), (0, 1))),
          256 / iter_shape[0])
c = zoom(cp.asnumpy(cp.flip(cp.abs(x), (0, 1))), 256 / sim_shape[0])

if phantom == 'brainweb':
    gm = np.flip(np.load(odir.parent / 'gm_256.npy'), (0, 1))
    wm = binary_erosion(np.flip(np.load(odir.parent / 'wm_256.npy'), (0, 1)),
                        iterations=3)

    IFFT_gm = a[gm == 1].mean()
    IFFT_wm = a[wm == 1].mean()

    ITER_gm = a2[gm == 1].mean()
    ITER_wm = a2[wm == 1].mean()

    AGR_1_gm = b[gm == 1].mean()
    AGR_1_wm = b[wm == 1].mean()

    AGR_b_gm = b3[gm == 1].mean()
    AGR_b_wm = b3[wm == 1].mean()

    true_gm = c[gm == 1].mean()
    true_wm = c[wm == 1].mean()

    rowlabels = [
        f'IFFT GM {IFFT_gm:.2f} WM {IFFT_wm:.2f} GM/WM {(IFFT_gm/IFFT_wm):.2f}',
        f'ITER GM {ITER_gm:.2f} WM {ITER_wm:.2f} GM/WM {(ITER_gm/ITER_wm):.2f}',
        f'AGR_1 GM {AGR_1_gm:.2f} WM {AGR_1_wm:.2f} GM/WM {(AGR_1_gm/AGR_1_wm):.2f}',
        f'AGR_B GM {AGR_b_gm:.2f} WM {AGR_b_wm:.2f} GM/WM {(AGR_b_gm/AGR_b_wm):.2f}',
        f'true GM {true_gm:.2f} WM {true_wm:.2f} GM/WM {(true_gm/true_wm):.2f}',
    ]
else:
    rowlabels = None

plt.rcParams.update({'font.size': 5})
vi = pv.ThreeAxisViewer([a, a2, b, b3, c],
                        sl_z=112,
                        sl_x=112,
                        ls='',
                        rowlabels=rowlabels,
                        imshow_kwargs=dict(vmin=0,
                                           vmax=1.1 * float(x.real.max()),
                                           cmap='Greys_r'))
vi.fig.savefig(odir / f'screenshot.png')

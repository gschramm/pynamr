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

from utils import setup_blob_phantom, setup_brainweb_phantom, kb_rolloff
from utils_sigpy import nufft_t2star_operator

#--------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--max_num_iter', type=int, default=500)
parser.add_argument('--noise_level', type=float, default=3e-2)
parser.add_argument('--phantom',
                    type=str,
                    default='brainweb',
                    choices=['brainweb', 'blob'])
parser.add_argument('--no_decay', action='store_true')
parser.add_argument('--sigma', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

max_num_iter = args.max_num_iter
noise_level = args.noise_level
phantom = args.phantom
no_decay = args.no_decay
sigma = args.sigma
seed = args.seed

cp.random.seed(seed)

#---------------------------------------------------------------
# fixed parameters

simshape = (160, 160, 160)
ishape = (64, 64, 64)

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

odir = Path(
    'run_ifft'
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
    x, t1_image, T2short_ms, T2long_ms = setup_blob_phantom(simshape[0],
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
kernel = 'kaiser_bessel'
width = 2
param = 9.14

data_echo_1_gridded = sigpy.gridding(data_echo_1,
                                     cp.asarray(k_1_cm.reshape(-1, 3)) *
                                     field_of_view_cm,
                                     ishape,
                                     kernel=kernel,
                                     width=width,
                                     param=param)
data_echo_2_gridded = sigpy.gridding(data_echo_2,
                                     cp.asarray(k_1_cm.reshape(-1, 3)) *
                                     field_of_view_cm,
                                     ishape,
                                     kernel=kernel,
                                     width=width,
                                     param=param)
samp_dens = sigpy.gridding(cp.ones_like(data_echo_1),
                           cp.asarray(k_1_cm.reshape(-1, 3)) *
                           field_of_view_cm,
                           ishape,
                           kernel=kernel,
                           width=width,
                           param=param)
ifft_op = sigpy.linop.IFFT(ishape)

data_echo_1_gridded_corr = data_echo_1_gridded.copy()
data_echo_1_gridded_corr[samp_dens > 0] /= samp_dens[samp_dens > 0]
data_echo_2_gridded_corr = data_echo_2_gridded.copy()
data_echo_2_gridded_corr[samp_dens > 0] /= samp_dens[samp_dens > 0]

# perform IFFT recon (without correction for fall-of due to k-space interpolation)
if phantom == 'brainweb':
    ifft_scale = 62.5
elif phantom == 'blob':
    ifft_scale = 80.5
else:
    raise ValueError

ifft1 = ifft_scale * ifft_op(data_echo_1_gridded_corr)
ifft2 = ifft_scale * ifft_op(data_echo_2_gridded_corr)

tmp_x = cp.linspace(-width / 2, width / 2, ishape[0])
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

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# recons without decay model and non-anatomical prior
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

outfile1 = odir / f'recon_echo_1_no_decay_model_{max_num_iter}.npz'

proxfc1 = sigpy.prox.L2Reg(data_echo_1.shape, 1, y=-data_echo_1)

u1 = cp.zeros(recon_operator.oshape, dtype=data_echo_1.dtype)

if not outfile1.exists():
    alg1 = sigpy.alg.PrimalDualHybridGradient(proxfc=proxfc1,
                                              proxg=sigpy.prox.NoOp(
                                                  recon_operator.ishape),
                                              A=recon_operator,
                                              AH=recon_operator.H,
                                              x=deepcopy(0 * ifft1),
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
    it_recon = alg1.x
else:
    d1 = cp.load(outfile1)
    it_recon = d1['x']
    u1 = d1['u']

#-------------------------------------------------------------------------------
# show results
from scipy.ndimage import zoom

a = zoom(cp.asnumpy(cp.flip(cp.abs(ifft1), (0, 1))), simshape[0] / ishape[0])
b = zoom(cp.asnumpy(cp.flip(cp.abs(it_recon), (0, 1))),
         simshape[0] / ishape[0])
c = cp.asnumpy(cp.flip(cp.abs(x), (0, 1)))

vi = pv.ThreeAxisViewer([a, b, c],
                        sl_z=73,
                        sl_x=73,
                        ls='',
                        rowlabels=[
                            'IFFT gridded data', 'iterative non-uniform data',
                            'ground truth'
                        ],
                        imshow_kwargs=dict(vmin=0,
                                           vmax=1.1 * float(x.real.max()),
                                           cmap='Greys_r'))

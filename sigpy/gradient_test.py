"""minimal script that shows how to solve L2square data fidelity + anatomical DTV prior"""

import argparse
import json
from pathlib import Path
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndimage
import sigpy
from scipy.optimize import fmin_l_bfgs_b

from utils import read_GE_ak_wav
from utils_sigpy import NUFFTT2starDualEchoModel, projected_gradient_operator
import pymirc.viewer as pv
from pymirc.image_operations import zoom3d

from gradient_test_utils import cost_and_grad_wrapper, setup_gradient_brainweb_phantom

#--------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--max_num_iter', type=int, default=200)
parser.add_argument('--noise_level', type=float, default=0)
parser.add_argument('--no_decay', action='store_true')
parser.add_argument('--sigma', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gradient_factor', type=int, default=1)
parser.add_argument('--no_decay_model', action='store_true')
args = parser.parse_args()

max_num_iter = args.max_num_iter
noise_level = args.noise_level
phantom = 'brainweb'
no_decay = args.no_decay
seed = args.seed
gradient_factor = args.gradient_factor
no_decay_model = args.no_decay_model

cp.random.seed(seed)

#---------------------------------------------------------------
# fixed parameters

# image shape for data simulation
sim_shape = (128*3, 128*3, 128*3)
num_sim_chunks = 512

# image shape for iterative recons
ishape = (128, 128, 128)
# grid shape for IFFTs
grid_shape = (64, 64, 64)

field_of_view_cm: float = 22.

# 10us sampling time
acq_sampling_time_ms: float = 0.016

# echo times in ms
echo_time_1_ms = 0.455
echo_time_2_ms = 5.

# signal fraction decaying with short T2* time
short_fraction = 0.6

time_bin_width_ms: float = 0.25

# read the data root directory from the config file
with open('.simulation_config.json', 'r') as f:
    data_root_dir: str = json.load(f)['data_root_dir']

odir = Path(data_root_dir) / 'gradient_test' / f'{phantom}_nodecay_{no_decay}_max_num_iter_{max_num_iter:04}'
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
    T2long_ms_gm: float = 20.
    T2long_ms_wm: float = 20.
    T2short_ms_csf: float = 50.
    T2short_ms_gm: float = 3.
    T2short_ms_wm: float = 3.

field_of_view_cm: float = 22.
phantom_data_path: Path = Path(data_root_dir) / 'brainweb54'

# (1) setup the brainweb phantom with the given simulation matrix size
x, t1_image, T2short_ms, T2long_ms, roi_image = setup_gradient_brainweb_phantom(
    sim_shape[0],
    phantom_data_path,
    field_of_view_cm=field_of_view_cm,
    T2long_ms_csf=T2long_ms_csf,
    T2long_ms_gm=T2long_ms_gm,
    T2long_ms_wm=T2long_ms_wm,
    T2short_ms_csf=T2short_ms_csf,
    T2short_ms_gm=T2short_ms_gm,
    T2short_ms_wm=T2short_ms_wm,
    csf_na_concentration=1.5,
    gm_na_concentration=0.6,
    wm_na_concentration=0.4,
    other_na_concentration=0.3,
    add_anatomical_mismatch=True,
    add_T2star_bias=False)


# move image to GPU
x = cp.asarray(x.astype(np.complex128))

true_ratio_image_short = cp.array(
    np.exp(-(echo_time_2_ms - echo_time_1_ms) / T2short_ms))
true_ratio_image_long = cp.array(
    np.exp(-(echo_time_2_ms - echo_time_1_ms) / T2long_ms))

np.save(odir / 'roi_image.npy', roi_image)
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
dt_us = acq_sampling_time_ms * 1000
# gamma by 2pi in MHz/T
gamma_by_2pi_MHz_T: float = 11.262

# kmax for 64 matrix (edge)
kmax64_1_cm = 1 / (2 * field_of_view_cm / 64)

# read gradients in T/m with shape (num_samples_per_readout, num_readouts, 3)
grads_T_m, bw, fov, desc, N, params = read_GE_ak_wav(
    Path(data_root_dir) / 'tpi_gradients/ak_grad56.wav')

# k values in 1/cm with shape (num_samples_per_readout, num_readouts, 3)
k_1_cm = 0.01 * np.cumsum(grads_T_m, axis=0) * dt_us * gamma_by_2pi_MHz_T

# simulate stronger gradients using every n-th readout point
k_1_cm = k_1_cm[::gradient_factor, ...]

k_1_cm_abs = np.linalg.norm(k_1_cm, axis=2)
print(f'readout kmax .: {k_1_cm_abs.max():.2f} 1/cm')
print(f'64 kmax      .: {kmax64_1_cm:.2f} 1/cm')

data_echo_1 = []
data_echo_2 = []

for i_chunk, k_inds in enumerate(
        np.array_split(np.arange(k_1_cm.shape[0]), num_sim_chunks)):
    print('simulating data chunk', i_chunk)

    data_model = NUFFTT2starDualEchoModel(
        sim_shape,
        k_1_cm[k_inds, ...],
        field_of_view_cm=field_of_view_cm,
        acq_sampling_time_ms=acq_sampling_time_ms,
        time_bin_width_ms=time_bin_width_ms,
        echo_time_1_ms=echo_time_1_ms + k_inds[0] *
        acq_sampling_time_ms,  # account of acq. offset time of every chunk
        echo_time_2_ms=echo_time_2_ms + k_inds[0] *
        acq_sampling_time_ms)  # account of acq. offset time of every chunk

    data_operator_1_short, data_operator_2_short = data_model.get_operators_w_decay_model(
        true_ratio_image_short)
    data_operator_1_long, data_operator_2_long = data_model.get_operators_w_decay_model(
        true_ratio_image_long)

    #--------------------------------------------------------------------------
    # simulate noise-free data
    data_echo_1.append(short_fraction * data_operator_1_short(x) +
                       (1 - short_fraction) * data_operator_1_long(x))
    data_echo_2.append(short_fraction * data_operator_2_short(x) +
                       (1 - short_fraction) * data_operator_2_long(x))

    del data_operator_1_short
    del data_operator_2_short
    del data_operator_1_long
    del data_operator_2_long
    del data_model

data_echo_1 = cp.concatenate(data_echo_1)
data_echo_2 = cp.concatenate(data_echo_2)

# add noise to the data
nl = noise_level * cp.abs(data_echo_1.max())
data_echo_1 += nl * (cp.random.randn(*data_echo_1.shape) +
                     1j * cp.random.randn(*data_echo_1.shape))
data_echo_2 += nl * (cp.random.randn(*data_echo_2.shape) +
                     1j * cp.random.randn(*data_echo_2.shape))

d1 = data_echo_1.reshape(k_1_cm.shape[:-1])
d2 = data_echo_2.reshape(k_1_cm.shape[:-1])

# print info related to SNR
for i in np.linspace(0, d1.shape[1], 10, endpoint=False).astype(int):
    print(
        f'{i:04} {float(cp.abs(d1[:, i]).max() / cp.abs(d1[-100:, i]).std()):.2f}'
    )

#--------------------------------------------------------------------
#--------------------------------------------------------------------
# adjoint NUFFT recon
#--------------------------------------------------------------------
#--------------------------------------------------------------------

# setup the density compensation weights
abs_k = np.linalg.norm(k_1_cm, axis=-1)
abs_k_twist = abs_k[114 // gradient_factor, 0]

dcf = cp.asarray(np.clip(abs_k**2, None, abs_k_twist**2)).ravel()

ifft1 = sigpy.nufft_adjoint(data_echo_1 * dcf,
                            cp.array(k_1_cm).reshape(-1, 3) * field_of_view_cm,
                            grid_shape)

ifft2 = sigpy.nufft_adjoint(data_echo_2 * dcf,
                            cp.array(k_1_cm).reshape(-1, 3) * field_of_view_cm,
                            grid_shape)

# interpolate to recons (128) grid
ifft1 = ndimage.zoom(ifft1,
                     ishape[0] / grid_shape[0],
                     order=1,
                     prefilter=False)
ifft2 = ndimage.zoom(ifft2,
                     ishape[0] / grid_shape[0],
                     order=1,
                     prefilter=False)

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

acq_model = NUFFTT2starDualEchoModel(ishape,
                                     k_1_cm,
                                     field_of_view_cm=field_of_view_cm,
                                     acq_sampling_time_ms=acq_sampling_time_ms,
                                     time_bin_width_ms=time_bin_width_ms,
                                     echo_time_1_ms=echo_time_1_ms,
                                     echo_time_2_ms=echo_time_2_ms)

# setup scaled single echo nufft operator
nufft_echo1_no_decay, nufft_echo2_no_decay = acq_model.get_operators_wo_decay_model(
)
del nufft_echo2_no_decay

#---------------------------------------------------------------------------------------
# LBFGS recons of data fidelity + quadratic prior

G = (1 / np.sqrt(12)) * sigpy.linop.FiniteDifference(ishape, axes=None)

if no_decay_model:
    A = nufft_echo1_no_decay
else:
    fwd_short_echo_1, fwd_short_echo_2 = acq_model.get_operators_w_decay_model(
        cp.asarray(zoom3d(cp.asnumpy(true_ratio_image_short), ishape[0]/sim_shape[0])))
    del fwd_short_echo_2
    
    fwd_long_echo_1, fwd_long_echo_2 = acq_model.get_operators_w_decay_model(
        cp.asarray(zoom3d(cp.asnumpy(true_ratio_image_long), ishape[0]/sim_shape[0])))
    del fwd_long_echo_2
    
    A = short_fraction*fwd_short_echo_1 + (1-short_fraction)*fwd_long_echo_1

x0 = cp.asnumpy(ifft1).view('(2,)float').ravel() / 10

betas = [0., 16., 32., 64., 128., 256., 512.]

recons = np.zeros((len(betas),) + ishape, dtype=np.complex128)

for ib, beta in enumerate(betas):
    res = fmin_l_bfgs_b(cost_and_grad_wrapper,
                        x0,
                        args=(A, data_echo_1, G, beta),
                        maxiter=max_num_iter,
                        disp=1)

    # convert flat pseudo complex array to complex
    recon = np.squeeze(res[0].view(dtype=np.complex128)).reshape(A.ishape)
    np.save(odir / f'recon_quad_prior_gf_{gradient_factor:02}_nl_{noise_level:.1E}_beta_{beta:.1E}_s_{seed:03}', recon)

    recons[ib,...] = recon

ims = dict(vmax = 9, cmap = 'Greys_r')
vi = pv.ThreeAxisViewer(np.abs(recons), imshow_kwargs = ims)

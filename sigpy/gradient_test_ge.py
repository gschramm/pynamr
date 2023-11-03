"""script for Na AGR of GE dual echo sodium data"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import cupyx.scipy.ndimage as ndimage
import scipy.ndimage as ndi
from scipy.optimize import fmin_l_bfgs_b
import sigpy
import pymirc.viewer as pv

from pathlib import Path

from utils import read_GE_ak_wav
from utils_sigpy import NUFFTT2starDualEchoModel
from gradient_test_utils import cost_and_grad_wrapper

import argparse

parser = argparse.ArgumentParser(
    description='AGR sodium recons for GE data',
    epilog=
    'DATA_DIR must contain the files: grad.wav, Pecho_1.7.h5, Pecho_2.7.h5, t1.nii'
)
parser.add_argument('--data_dir',
                    type=str,
                    default='/data/sodium_mr/20230316_MR3_GS_QED/pfiles/g16')
parser.add_argument('--max_num_iter', type=int, default=200)
parser.add_argument('--odir', type=str, default=None)
parser.add_argument('--matrix_size', type=int, default=128)
parser.add_argument('--clip_kmax', action='store_true')
args = parser.parse_args()

#--------------------------------------------------------------------
# input parameters
data_dir = Path(args.data_dir)
max_num_iter = args.max_num_iter
matrix_size = args.matrix_size
clip_kmax = args.clip_kmax

#--------------------------------------------------------------------
# fixed parameters
gradient_file = data_dir / 'grad.wav'
echo_1_data_file = data_dir / 'Pecho_1.7.h5'
echo_2_data_file = data_dir / 'Pecho_2.7.h5'
t1_file = data_dir / 't1.nii'

# shape of the images to be reconstructed
ishape = (matrix_size, matrix_size, matrix_size)

# time bin width for T2* decay modeling
time_bin_width_ms: float = 0.25

# echo times in ms
echo_time_1_ms = 0.455
echo_time_2_ms = 5.

# show the readout data (to spot bad readouts)
show_readouts = False

# create the output directory
if args.odir is None:
    odir = Path(echo_1_data_file).parent / f'grad_test_recons_{matrix_size}'
else:
    odir = Path(args.odir)
odir.mkdir(exist_ok=True, parents=True)

#--------------------------------------------------------------------
#--------------------------------------------------------------------
# read the gradient file to get the k-space trajectory
#--------------------------------------------------------------------
#--------------------------------------------------------------------

# read gradients in T/m with shape (num_samples_per_readout, num_readouts, 3)
grads_T_m, bw, fov, desc, N, params = read_GE_ak_wav(gradient_file)

# time sampling step in micro seconds
dt_us = params[7]
acq_sampling_time_ms = dt_us * 1e-3

# gamma by 2pi in MHz/T for 23Na
gamma_by_2pi_MHz_T: float = 11.262

# k values in 1/cm with shape (num_samples_per_readout, num_readouts, 3)
k_1_cm = 0.01 * np.cumsum(grads_T_m, axis=0) * dt_us * gamma_by_2pi_MHz_T

# get the FOV in cm, the value from the header has to be corrected by
# the ratio of gamma(1H) / gamma(23Na)
#field_of_view_cm = 100 * fov * 42.577 / gamma_by_2pi_MHz_T
field_of_view_cm = 22.

#--------------------------------------------------------------------
#--------------------------------------------------------------------
# read the acquired data
#--------------------------------------------------------------------
#--------------------------------------------------------------------

with h5py.File(echo_1_data_file) as data1:
    data_echo_1 = data1['/data'][:]
    rotation = int(data1['/header']['rdb_hdr']['rotation'][0])

with h5py.File(echo_2_data_file) as data2:
    data_echo_2 = data2['/data'][:]

# convert data to complex (from two reals) send data arrays to GPU
data_echo_1 = cp.asarray(data_echo_1['real'] + 1j * data_echo_1['imag'])
data_echo_2 = cp.asarray(data_echo_2['real'] + 1j * data_echo_2['imag'])

if k_1_cm.shape[0] < data_echo_1.shape[0]:
    data_echo_1 = data_echo_1[:k_1_cm.shape[0], :]
    data_echo_2 = data_echo_2[:k_1_cm.shape[0], :]

# calculate the signal max across all readouts to detect potential readout problems
max_echo_1 = cp.asnumpy(cp.abs(data_echo_1).max(0))
max_echo_2 = cp.asnumpy(cp.abs(data_echo_2).max(0))

# calculate the most common maximum signal to find outliers
h1 = np.histogram(ndi.gaussian_filter1d(max_echo_1, 5), bins=100)
most_common_max_1 = h1[1][np.argmax(h1[0])]
h2 = np.histogram(ndi.gaussian_filter1d(max_echo_2, 5), bins=100)
most_common_max_2 = h2[1][np.argmax(h2[0])]

th1 = most_common_max_1 - 3 * max_echo_1[max_echo_1 >= 0.95 *
                                         most_common_max_1].std()
th2 = most_common_max_2 - 3 * max_echo_2[max_echo_2 >= 0.95 *
                                         most_common_max_2].std()

i_bad_1 = np.where(max_echo_1 < th1)[0]
i_bad_2 = np.where(max_echo_2 < th2)[0]

print(f'num bad readouts 1st echo {i_bad_1.size}')
print(f'num bad readouts 2nd echo {i_bad_2.size}')

# setup the data weights (1 for good reaouts, 0 for bad readouts)
data_weights_1 = cp.ones(data_echo_1.shape, dtype=np.uint8)
data_weights_2 = cp.ones(data_echo_2.shape, dtype=np.uint8)

data_weights_1[:, i_bad_1] = 0
data_weights_2[:, i_bad_2] = 0
# also ignore the first readout point
data_weights_1[0, :] = 0
data_weights_2[0, :] = 0

# set data bins of bad readouts to 0
data_echo_1 *= data_weights_1
data_echo_2 *= data_weights_2

# normalize the data such that max is 1373 (similar to brainweb simulation)
norm_fac = 1373. / np.abs(data_echo_1).max()
data_echo_1 *= norm_fac
data_echo_2 *= norm_fac

# print info related to SNR
for i in np.linspace(0, data_echo_1.shape[1], 10, endpoint=False).astype(int):
    print(
        f'{i:04} {float(cp.abs(data_echo_1[:, i]).max() / cp.abs(data_echo_1[-100:, i]).std()):.2f}'
    )


# ignore last acquired points that are added by the scanner
if data_echo_1.shape[0] > k_1_cm.shape[0]:
    data_echo_1 = data_echo_1[:k_1_cm.shape[0], :]
if data_echo_2.shape[0] > k_1_cm.shape[0]:
    data_echo_2 = data_echo_2[:k_1_cm.shape[0], :]

print(f'acquisition sampling time .: {dt_us:.2f} us')
print(f'field of view             .: {field_of_view_cm:.2f} cm')
print(
    f'Gmax                      .: {100*np.linalg.norm(grads_T_m, axis=-1).max():.2f} G/cm'
)
print(
    f'kmax                      .: {np.linalg.norm(k_1_cm, axis=-1).max():.2f} 1/cm'
)
print(
    f'readout time              .: {(k_1_cm.shape[0]+1) * acq_sampling_time_ms:.2f} ms'
)
print(f'rotation                  .: {rotation}')

#--------------------------------------------------------------------
#--------------------------------------------------------------------
# regrid the data and do simple IFFT
#--------------------------------------------------------------------
#--------------------------------------------------------------------

grid_shape = (64,64,64)

# setup the density compensation weights
abs_k = np.linalg.norm(k_1_cm, axis=-1)

# neglect all data points above 1.4759 1/cm
if clip_kmax:
    clip_suffix = '_clipped_kmax'
    imax = np.where(abs_k[:,0] <= 1.4759)[0].max() + 1
    abs_k = abs_k[:imax,:]
    k_1_cm = k_1_cm[:imax,:,:]
    data_echo_1 = data_echo_1[:imax,:]
    data_echo_2 = data_echo_2[:imax,:]
    data_weights_1 = data_weights_1[:imax,:]
    data_weights_2 = data_weights_2[:imax,:]
else:
    clip_suffix = ''

# calculate where the linear readout ends
dk = abs_k[1:,0] - abs_k[:-1,0]
ilin = np.where(dk > 0.99*dk.max())[0].max()
abs_k_twist = abs_k[ilin, 0]
dcf = cp.asarray(np.clip(abs_k**2, None, abs_k_twist**2)).ravel()

# flatten the data for the nufft operators
data_echo_1 = data_echo_1.ravel()
data_echo_2 = data_echo_2.ravel()
data_weights_1 = data_weights_1.ravel()
data_weights_2 = data_weights_2.ravel()

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

cp.save(odir / f'nufft_adjoint_128_gf{clip_suffix}.npy', ifft1)

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# setup (unscaled) acquisition model
acq_model = NUFFTT2starDualEchoModel(ishape,
                                     k_1_cm,
                                     field_of_view_cm=field_of_view_cm,
                                     acq_sampling_time_ms=acq_sampling_time_ms,
                                     time_bin_width_ms=time_bin_width_ms,
                                     echo_time_1_ms=echo_time_1_ms,
                                     echo_time_2_ms=echo_time_2_ms)


nufft_echo1_no_decay, nufft_echo2_no_decay = acq_model.get_operators_wo_decay_model()

G = (1 / np.sqrt(12)) * sigpy.linop.FiniteDifference(ishape, axes=None)

decay_suffix = '_no_decay_model'
A = sigpy.linop.Multiply(nufft_echo1_no_decay.oshape, data_weights_1) * nufft_echo1_no_decay

x0 = cp.asnumpy(ifft1).view('(2,)float').ravel() / 10

if matrix_size == 128:
    betas = [0., 64., 512., 1024., 2048., 4096.]
else:
    betas = [0., 32., 64., 128., 256., 512., 1024.]

recons = np.zeros((len(betas),) + ishape, dtype=np.complex128)

for ib, beta in enumerate(betas):

    ofile = odir / f'recon_quad_prior_beta_{beta:.1E}{decay_suffix}{clip_suffix}.npy'

    if not ofile.exists():
        res = fmin_l_bfgs_b(cost_and_grad_wrapper,
                            x0,
                            args=(A, data_echo_1, G, beta),
                            maxiter=max_num_iter,
                            disp=1)

        # convert flat pseudo complex array to complex
        recon = np.squeeze(res[0].view(dtype=np.complex128)).reshape(A.ishape)
        np.save(ofile, recon)
    else:
        recon = np.load(ofile)

    recons[ib,...] = recon

ims = dict(vmax = 9 * (128/matrix_size)**1.5, cmap = 'Greys_r')
vi = pv.ThreeAxisViewer(np.abs(recons), imshow_kwargs = ims)

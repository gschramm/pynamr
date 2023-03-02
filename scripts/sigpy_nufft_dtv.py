"""minimal script that shows how to solve L2square data fidelity + anatomical DTV prior"""
import argparse
import sigpy
import math
import cupy as cp
import numpy as np

import pymirc.viewer as pv

import json
from pathlib import Path
from utils import setup_blob_phantom, setup_brainweb_phantom, read_tpi_gradient_files

import cupyx.scipy.ndimage as ndimage

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
parser.add_argument('--beta', type=float, default=1e-2)
parser.add_argument('--max_num_iter', type=int, default=300)
parser.add_argument('--gradient_strength',
                    type=int,
                    default=16,
                    choices=[16, 24, 32])
parser.add_argument('--noise_level', type=float, default=1e-2)
parser.add_argument('--phantom',
                    type=str,
                    default='brainweb',
                    choices=['brainweb', 'blob'])
parser.add_argument('--no_decay', action='store_true')
args = parser.parse_args()

regularization_operator = args.regularization_operator
regularization_norm = args.regularization_norm
beta = args.beta
max_num_iter = args.max_num_iter
noise_level = args.noise_level
gradient_strength: int = args.gradient_strength
phantom = args.phantom
no_decay: bool = args.no_decay

ishape = (128, 128, 128)
sigma = 1e-1

field_of_view_cm: float = 22.

# 10us sampling time
acq_sampling_time_ms: float = 0.01

# echo times in ms
echo_time_1_ms = 0.5
echo_time_2_ms = 5.

# scaling factor for nufft operators such that their norm is approx. equal to gradient op.
scale = 0.03

# signal fraction decaying with short T2* time
short_fraction = 0.6

# read the data root directory from the config file
with open('.simulation_config.json', 'r') as f:
    data_root_dir: str = json.load(f)['data_root_dir']

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

simulation_matrix_size: int = ishape[0]
field_of_view_cm: float = 22.
phantom_data_path: Path = Path(data_root_dir) / 'brainweb54'

# (1) setup the brainweb phantom with the given simulation matrix size
if phantom == 'brainweb':
    x, t1_image, T2short_ms, T2long_ms = setup_brainweb_phantom(
        simulation_matrix_size,
        phantom_data_path,
        field_of_view_cm=field_of_view_cm,
        T2long_ms_csf=T2long_ms_csf,
        T2long_ms_gm=T2long_ms_gm,
        T2long_ms_wm=T2long_ms_wm,
        T2short_ms_csf=T2short_ms_csf,
        T2short_ms_gm=T2short_ms_gm,
        T2short_ms_wm=T2short_ms_wm)
elif phantom == 'blob':
    x, t1_image, T2short_ms, T2long_ms = setup_blob_phantom(
        simulation_matrix_size)
else:
    raise ValueError

# move image to GPU
x = cp.asarray(x.astype(np.complex128))

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# data simulation block
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

if gradient_strength == 16:
    gradient_file: str = str(
        Path(data_root_dir) / 'tpi_gradients/n28p4dt10g16_23Na_v1')
elif gradient_strength == 24:
    gradient_file: str = str(
        Path(data_root_dir) / 'tpi_gradients/n28p4dt10g24f23')
elif gradient_strength == 32:
    gradient_file: str = str(
        Path(data_root_dir) / 'tpi_gradients/n28p4dt10g32f23')
elif gradient_strength == 48:
    gradient_file: str = str(
        Path(data_root_dir) / 'tpi_gradients/n28p4dt10g48f23')
else:
    raise ValueError

kmax = 1 / (2 * field_of_view_cm / 64)

kx, ky, kz, header, n_readouts_per_cone = read_tpi_gradient_files(
    gradient_file)

true_ratio_image_short = cp.array(
    np.exp(-(echo_time_2_ms - echo_time_1_ms) / T2short_ms))
true_ratio_image_long = cp.array(
    np.exp(-(echo_time_2_ms - echo_time_1_ms) / T2long_ms))

# setup the data operators for the 1/2 echo using the short T2* time
data_operator_1_short, data_operator_2_short = nufft_t2star_operator(
    ishape,
    kx,
    ky,
    kz,
    field_of_view_cm=field_of_view_cm,
    acq_sampling_time_ms=acq_sampling_time_ms,
    time_bin_width_ms=0.25,
    scale=scale,
    add_mirrored_coordinates=True,
    echo_time_1_ms=echo_time_1_ms,
    echo_time_2_ms=echo_time_2_ms,
    ratio_image=true_ratio_image_short)

# setup the data operators for the 1/2 echo using the long T2* time
data_operator_1_long, data_operator_2_long = nufft_t2star_operator(
    ishape,
    kx,
    ky,
    kz,
    field_of_view_cm=field_of_view_cm,
    acq_sampling_time_ms=acq_sampling_time_ms,
    time_bin_width_ms=0.25,
    scale=scale,
    add_mirrored_coordinates=True,
    echo_time_1_ms=echo_time_1_ms,
    echo_time_2_ms=echo_time_2_ms,
    ratio_image=true_ratio_image_long)

#--------------------------------------------------------------------------
# simulate noise-free data
data_echo_1 = short_fraction * data_operator_1_short(x) + (
    1 - short_fraction) * data_operator_1_long(x)
data_echo_2 = short_fraction * data_operator_2_short(x) + (
    1 - short_fraction) * data_operator_2_long(x)

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

recon_operator = nufft_t2star_operator(
    ishape,
    kx,
    ky,
    kz,
    field_of_view_cm=field_of_view_cm,
    acq_sampling_time_ms=acq_sampling_time_ms,
    time_bin_width_ms=0.25,
    scale=scale,
    add_mirrored_coordinates=True,
    echo_time_1_ms=echo_time_1_ms,
    echo_time_2_ms=echo_time_2_ms)

# grid data for conventional IFFT recon
data_echo_1_gridded = sigpy.gridding(data_echo_1,
                                     recon_operator.linops[1].coord,
                                     ishape,
                                     kernel='spline',
                                     width=1)
data_echo_2_gridded = sigpy.gridding(data_echo_2,
                                     recon_operator.linops[1].coord,
                                     ishape,
                                     kernel='spline',
                                     width=1)
samp_dens = sigpy.gridding(cp.ones_like(data_echo_1),
                           recon_operator.linops[1].coord,
                           ishape,
                           kernel='kaiser_bessel')
ifft_op = sigpy.linop.IFFT(ishape)

data_echo_1_gridded_corr = data_echo_1_gridded.copy()
data_echo_1_gridded_corr[samp_dens > 0] /= samp_dens[samp_dens > 0]
data_echo_2_gridded_corr = data_echo_2_gridded.copy()
data_echo_2_gridded_corr[samp_dens > 0] /= samp_dens[samp_dens > 0]

# perform IFFT recon (without correction for fall-of due to k-space interpolation)
ifft_scale = 1648.5
ifft1 = ifft_scale * ifft_op(data_echo_1_gridded_corr)
ifft2 = ifft_scale * ifft_op(data_echo_2_gridded_corr)

#vi = pv.ThreeAxisViewer([np.abs(cp.asnumpy(ifft1)), np.abs(cp.asnumpy(ifft2))])

del data_operator_1_short
del data_operator_1_long
del data_operator_2_short
del data_operator_2_long

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# setup projected gradient operator for DTV

# set up the operator for regularization
G = sigpy.linop.FiniteDifference(ishape, axes=None)

# setup a joint gradient field
prior_image = cp.asarray(t1_image)
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

if regularization_operator == 'projected_gradient':
    R = PG
elif regularization_operator == 'gradient':
    R = G
else:
    raise ValueError('unknown regularization operator')

if regularization_norm == 'L2':
    proxg = sigpy.prox.L2Reg(R.oshape, lamda=beta)
elif regularization_norm == 'L1':
    proxg = sigpy.prox.L1Reg(R.oshape, lamda=beta)
else:
    raise ValueError('unknown regularization norm')

#--------------------------------------------------------------------------
# run iterative recon of first echo with prior but without T2* decay modeling
x0 = ifft1.copy()
alg1 = sigpy.app.LinearLeastSquares(recon_operator,
                                    data_echo_1,
                                    x=x0,
                                    G=R,
                                    proxg=proxg,
                                    sigma=sigma,
                                    max_iter=max_num_iter,
                                    max_power_iter=10)

recon_echo_1_wo_decay_model = alg1.run()
r1 = np.abs(cp.asnumpy(recon_echo_1_wo_decay_model))

x0 = ifft2.copy()
alg2 = sigpy.app.LinearLeastSquares(recon_operator,
                                    data_echo_2,
                                    x=x0,
                                    G=R,
                                    proxg=proxg,
                                    sigma=sigma,
                                    max_iter=max_num_iter,
                                    max_power_iter=10)

recon_echo_2_wo_decay_model = alg2.run()
r2 = np.abs(cp.asnumpy(recon_echo_2_wo_decay_model))

##-------------------------------------------------------------------------
## calculate the ratio between the two recons without T2* decay modeling
## to estimate a monoexponential T2*
#
#ratio = cp.clip(
#    cp.abs(recon_echo_2_no_decay) / cp.abs(recon_echo_1_no_decay), 0, 1)
## set ratio to one in voxels where there is low signal in the first echo
#mask = 1 - (cp.abs(recon_echo_1_no_decay) <
#            0.05 * cp.abs(recon_echo_1_no_decay).max())
#
#label, num_label = ndimage.label(mask == 1)
#size = np.bincount(label.ravel())
#biggest_label = size[1:].argmax() + 1
#clump_mask = label == biggest_label
#
#ratio[clump_mask == 0] = 1
#
##-------------------------------------------------------------------------
## setup operators including the estimated T2* decay modeling
#C1s = []
#C2s = []
#
#for i, time_bin_inds in enumerate(time_bins_inds):
#
#    #setup the decay image
#    readout_time_1_ms = echo_time_1_ms + time_bin_inds.mean() * sampling_time_ms
#    readout_time_2_ms = echo_time_2_ms + time_bin_inds.mean() * sampling_time_ms
#    decay_image_1 = ratio**((readout_time_1_ms) /
#                            (echo_time_2_ms - echo_time_1_ms))
#    decay_image_2 = ratio**((readout_time_2_ms) /
#                            (echo_time_2_ms - echo_time_1_ms))
#
#    k0 = cp.asarray(kx[:, time_bin_inds])
#    k1 = cp.asarray(ky[:, time_bin_inds])
#    k2 = cp.asarray(kz[:, time_bin_inds])
#
#    # the gradient files only contain a half sphere
#    # we add the 2nd half where all gradients are reversed
#    k0 = np.vstack((k0, -k0))
#    k1 = np.vstack((k1, -k1))
#    k2 = np.vstack((k2, -k2))
#
#    # reshape kx, ky, kz into single coordinate array
#    coords = cp.zeros((k0.size, 3))
#    coords[:, 0] = cp.asarray(k0.ravel())
#    coords[:, 1] = cp.asarray(k1.ravel())
#    coords[:, 2] = cp.asarray(k2.ravel())
#    # sigpy needs unitless k-space points (ranging from -n/2 ... n/2)
#    # -> we have to multiply with the field_of_view
#    coords *= field_of_view_cm
#
#    C1s.append(
#        (0.03) * sigpy.linop.NUFFT(ishape, coords, oversamp=1.25, width=4) *
#        sigpy.linop.Multiply(ishape, decay_image_1))
#
#    C2s.append(
#        (0.03) * sigpy.linop.NUFFT(ishape, coords, oversamp=1.25, width=4) *
#        sigpy.linop.Multiply(ishape, decay_image_2))
#
## operator including T2* decay for first echo
#C1 = sigpy.linop.Vstack(C1s)
#C2 = sigpy.linop.Vstack(C2s)
#
##-------------------------------------------------------------------------
## run a recon with T2* decay modeling
## recon of 1st echo
#x0 = recon_echo_1_no_decay.copy()
#reconstructor_echo1 = sigpy.app.LinearLeastSquares(C1,
#                                                   y1,
#                                                   x=x0,
#                                                   G=R,
#                                                   proxg=proxg,
#                                                   sigma=sigma,
#                                                   max_iter=max_num_iter,
#                                                   max_power_iter=10)
#
#recon_echo_1 = reconstructor_echo1.run()
#
#f1 = cp.asnumpy(recon_echo_1)
#
##-------------------------------------------------------------------------
## run a recon with T2* decay modeling using both echos
## recon of 1st echo
#
#D = sigpy.linop.Vstack([C1, C2])
#
#x0 = recon_echo_1_no_decay.copy()
#reconstructor_both_echos = sigpy.app.LinearLeastSquares(D,
#                                                        cp.concatenate(
#                                                            (y1, y2)),
#                                                        x=x0,
#                                                        G=R,
#                                                        proxg=proxg,
#                                                        sigma=sigma,
#                                                        max_iter=max_num_iter,
#                                                        max_power_iter=10)
#
#recon_both_echos = reconstructor_both_echos.run()
#
#g = cp.asnumpy(recon_both_echos)
#
#ims = 2 * [dict(vmin=0, vmax=3.5)] + [dict(vmin=0, vmax=1.)]
#
#vi = pv.ThreeAxisViewer(
#    [np.abs(e1), np.abs(g), np.abs(cp.asnumpy(ratio))], imshow_kwargs=ims)
#
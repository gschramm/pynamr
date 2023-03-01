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
args = parser.parse_args()

regularization_operator = args.regularization_operator
regularization_norm = args.regularization_norm
beta = args.beta
max_num_iter = args.max_num_iter
noise_level = args.noise_level
gradient_strength: int = args.gradient_strength

ishape = (128, 128, 128)
sigma = 1e-1

phantom = 'brainweb'

field_of_view_cm: float = 22.
no_decay: bool = True

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
# setup nufft operator

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

all_coords = []

# split the acquisition into chunks of 0.5ms = 500us (sampling is on 10us -> 50 elements per chunck)
time_bins_inds = np.array_split(np.arange(kx.shape[1]),
                                math.ceil(kx.shape[1] / 50))

As = []

for i, time_bin_inds in enumerate(time_bins_inds):
    k0 = cp.asarray(kx[:, time_bin_inds])
    k1 = cp.asarray(ky[:, time_bin_inds])
    k2 = cp.asarray(kz[:, time_bin_inds])

    # the gradient files only contain a half sphere
    # we add the 2nd half where all gradients are reversed
    k0 = np.vstack((k0, -k0))
    k1 = np.vstack((k1, -k1))
    k2 = np.vstack((k2, -k2))

    # reshape kx, ky, kz into single coordinate array
    coords = cp.zeros((k0.size, 3))
    coords[:, 0] = cp.asarray(k0.ravel())
    coords[:, 1] = cp.asarray(k1.ravel())
    coords[:, 2] = cp.asarray(k2.ravel())
    # sigpy needs unitless k-space points (ranging from -n/2 ... n/2)
    # -> we have to multiply with the field_of_view
    coords *= field_of_view_cm

    all_coords.append(coords)

    As.append(
        (0.03) * sigpy.linop.NUFFT(ishape, coords, oversamp=1.25, width=4))

all_coords = cp.vstack(all_coords)
A = sigpy.linop.Vstack(As)

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
# simulate noise-free data
y = A.apply(x)

# add noise to the data
y += noise_level * cp.abs(
    y.max()) * (cp.random.randn(*y.shape) + 1j * cp.random.randn(*y.shape))

# grid data for reference recon
y_gridded = sigpy.gridding(y, all_coords, ishape, kernel='spline', width=1)
samp_dens = sigpy.gridding(cp.ones_like(y),
                           all_coords,
                           ishape,
                           kernel='kaiser_bessel')
ifft_op = sigpy.linop.IFFT(ishape)

y_gridded_corr = y_gridded.copy()
y_gridded_corr[samp_dens > 0] /= samp_dens[samp_dens > 0]

# perform IFFT recon (without correction for fall-of due to k-space interpolation)
ifft = ifft_op(y_gridded_corr)
ifft *= (x.max() / cp.abs(ifft).max())

vi = pv.ThreeAxisViewer(np.abs(cp.asnumpy(ifft)))

##--------------------------------------------------------------------------
## iterative recon with structural prior
#x0 = ifft.copy()
#alg = sigpy.app.LinearLeastSquares(A,
#                                   y,
#                                   x=A.H(y),
#                                   G=R,
#                                   proxg=proxg,
#                                   sigma=sigma,
#                                   max_iter=max_num_iter,
#                                   max_power_iter=10)
#
#print(alg.sigma, alg.tau, alg.sigma * alg.tau)
#
#x_hat = alg.run()
#
#x_hat_cpu = cp.asnumpy(x_hat)
#
#np.save(
#    Path('run') /
#    f'{regularization_operator}_{regularization_norm}_{beta:.2e}_{max_num_iter:04}.npy',
#    x_hat_cpu,
#)
#
#vi = pv.ThreeAxisViewer([
#    np.abs(cp.asnumpy(ifft)),
#    np.abs(x_hat_cpu), x_hat_cpu.real, x_hat_cpu.imag
#])
#
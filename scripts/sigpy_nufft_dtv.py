"""minimal script that shows how to solve L2square data fidelity + anatomical DTV prior"""
import sigpy
import cupy as cp
import numpy as np

import pymirc.viewer as pv

import json
from pathlib import Path
from utils import setup_blob_phantom, setup_brainweb_phantom, read_tpi_gradient_files

#--------------------------------------------------------------------------

ishape = (128, 128, 128)
noise_level = 0.01
max_num_iter = 100
sigma = 1e-1

phantom = 'brainweb'

regularization_operator = 'projected_gradient'  # projected_gradient or gradient
regularization_norm = 'L1'  # L1 or L2
beta = 1e-2  # ca 1e-2 for L1 and 3e-1 for L2

field_of_view_cm: float = 22.
no_decay: bool = True
gradient_strength: int = 16

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
x = cp.asarray(x)

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

# the gradient files only contain a half sphere
# we add the 2nd half where all gradients are reversed
kx = np.vstack((kx, -kx))
ky = np.vstack((ky, -ky))
kz = np.vstack((kz, -kz))

# reshape kx, ky, kz into single coordinate array
coords = cp.zeros((kx.size, 3))
coords[:, 0] = cp.asarray(kx.ravel())
coords[:, 1] = cp.asarray(ky.ravel())
coords[:, 2] = cp.asarray(kz.ravel())
# sigpy needs unitless k-space points (ranging from -n/2 ... n/2)
# -> we have to multiply with the field_of_view
coords *= field_of_view_cm

A = (0.03) * sigpy.linop.NUFFT(ishape, coords, oversamp=1.25, width=4)
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
y_gridded = sigpy.gridding(y, coords, ishape, kernel='spline', width=1)
samp_dens = sigpy.gridding(cp.ones_like(y),
                           coords,
                           ishape,
                           kernel='kaiser_bessel')
ifft_op = sigpy.linop.IFFT(ishape)

y_gridded_corr = y_gridded.copy()
y_gridded_corr[samp_dens > 0] /= samp_dens[samp_dens > 0]

# perform IFFT recon (without correction for fall-of due to k-space interpolation)
ifft = ifft_op(y_gridded_corr)
ifft *= (x.max() / cp.abs(ifft).max())

#--------------------------------------------------------------------------
# iterative recon with structural prior
x0 = ifft.copy()
alg = sigpy.app.LinearLeastSquares(A,
                                   y,
                                   x=A.H(y),
                                   G=R,
                                   proxg=proxg,
                                   sigma=sigma,
                                   max_iter=max_num_iter)

print(alg.sigma, alg.tau, alg.sigma * alg.tau)

x_hat = alg.run()

x_hat_cpu = cp.asnumpy(x_hat)
vi = pv.ThreeAxisViewer([
    np.abs(cp.asnumpy(ifft)),
    np.abs(x_hat_cpu), x_hat_cpu.real, x_hat_cpu.imag
])

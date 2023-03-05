"""minimal script that shows how to solve L2square data fidelity + anatomical DTV prior"""
#TODO:
# - simulate data on higher grid
# - add anatomical mismatches

import argparse
import json
from pathlib import Path
from copy import deepcopy
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndimage
import sigpy

import pymirc.viewer as pv
from pymirc.image_operations import zoom3d

from utils import setup_blob_phantom, setup_brainweb_phantom, read_tpi_gradient_files
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
parser.add_argument('--beta', type=float, default=3e-3)
parser.add_argument('--max_num_iter', type=int, default=300)
parser.add_argument('--gradient_strength',
                    type=int,
                    default=16,
                    choices=[16, 24, 32])
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
gradient_strength: int = args.gradient_strength
phantom = args.phantom
no_decay = args.no_decay
sigma = args.sigma
seed = args.seed

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
scale = 0.005

# signal fraction decaying with short T2* time
short_fraction = 0.6

time_bin_width_ms: float = 0.25

# read the data root directory from the config file
with open('.simulation_config.json', 'r') as f:
    data_root_dir: str = json.load(f)['data_root_dir']

odir = Path(
    'run'
) / f'ro_{regularization_operator}_rn_{regularization_norm}_b_{beta:.1E}_i_{max_num_iter:04}_nl_{noise_level:.1E}_g_{gradient_strength}'
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

cp.save(odir / 'na_gt.npy', x)
cp.save(odir / 't1.npy', t1_image)
cp.save(odir / 'true_ratio_short.npy', true_ratio_image_short)
cp.save(odir / 'true_ratio_long.npy', true_ratio_image_long)

# setup the data operators for the 1/2 echo using the short T2* time
data_operator_1_short, data_operator_2_short = nufft_t2star_operator(
    simshape,
    kx,
    ky,
    kz,
    field_of_view_cm=field_of_view_cm,
    acq_sampling_time_ms=acq_sampling_time_ms,
    time_bin_width_ms=time_bin_width_ms,
    scale=scale,
    add_mirrored_coordinates=True,
    echo_time_1_ms=echo_time_1_ms,
    echo_time_2_ms=echo_time_2_ms,
    ratio_image=true_ratio_image_short)

# setup the data operators for the 1/2 echo using the long T2* time
data_operator_1_long, data_operator_2_long = nufft_t2star_operator(
    simshape,
    kx,
    ky,
    kz,
    field_of_view_cm=field_of_view_cm,
    acq_sampling_time_ms=acq_sampling_time_ms,
    time_bin_width_ms=time_bin_width_ms,
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

recon_operator = nufft_t2star_operator(
    ishape,
    kx,
    ky,
    kz,
    field_of_view_cm=field_of_view_cm,
    acq_sampling_time_ms=acq_sampling_time_ms,
    time_bin_width_ms=time_bin_width_ms,
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
ifft_scale = 19640.7 / 3.5
ifft1 = ifft_scale * ifft_op(data_echo_1_gridded_corr)
ifft2 = ifft_scale * ifft_op(data_echo_2_gridded_corr)

#vi = pv.ThreeAxisViewer([np.abs(cp.asnumpy(ifft1)), np.abs(cp.asnumpy(ifft2))])

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
A = sigpy.linop.Vstack([recon_operator, R])

#------------------------------------------------------
# reconstruct the first echo without T2* decay modeling

proxfc1 = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_1.shape, 1, y=-data_echo_1),
    sigpy.prox.Conj(proxg)
])
u1 = cp.zeros(A.oshape, dtype=data_echo_1.dtype)

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

#-----------------------------------------------------

proxfc2 = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_1.shape, 1, y=-data_echo_2),
    sigpy.prox.Conj(proxg)
])
u2 = cp.zeros(A.oshape, dtype=data_echo_2.dtype)

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

del A
del recon_operator

cp.save(odir / 'recon_echo_1_wo_decay_model.npy', recon_echo_1_wo_decay_model)
cp.save(odir / 'recon_echo_2_wo_decay_model.npy', recon_echo_2_wo_decay_model)

#-------------------------------------------------------------------------
# calculate the ratio between the two recons without T2* decay modeling
# to estimate a monoexponential T2*

est_ratio = cp.clip(
    cp.abs(recon_echo_2_wo_decay_model) / cp.abs(recon_echo_1_wo_decay_model),
    0, 1)
# set ratio to one in voxels where there is low signal in the first echo
mask = 1 - (cp.abs(recon_echo_1_wo_decay_model) <
            0.05 * cp.abs(recon_echo_1_wo_decay_model).max())

label, num_label = ndimage.label(mask == 1)
size = np.bincount(label.ravel())
biggest_label = size[1:].argmax() + 1
clump_mask = (label == biggest_label)

est_ratio[clump_mask == 0] = 1

cp.save(odir / 'est_ratio.npy', est_ratio)
#-------------------------------------------------------------------------
# setup the recon operators for the 1/2 echo using the estimated short T2* time (ratio)
recon_operator_1, recon_operator_2 = nufft_t2star_operator(
    ishape,
    kx,
    ky,
    kz,
    field_of_view_cm=field_of_view_cm,
    acq_sampling_time_ms=acq_sampling_time_ms,
    time_bin_width_ms=time_bin_width_ms,
    scale=scale,
    add_mirrored_coordinates=True,
    echo_time_1_ms=echo_time_1_ms,
    echo_time_2_ms=echo_time_2_ms,
    ratio_image=est_ratio)

#-------------------------------------------------------------------------
# redo the independent recons with updated operators including T2* modeling

A1 = sigpy.linop.Vstack([recon_operator_1, R])

alg11 = sigpy.alg.PrimalDualHybridGradient(
    proxfc=proxfc1,
    proxg=sigpy.prox.NoOp(A1.ishape),
    A=A1,
    AH=A1.H,
    x=deepcopy(recon_echo_1_wo_decay_model),
    u=u1,
    tau=1. / sigma,
    sigma=sigma,
    max_iter=max_num_iter)

print('recon echo 1 - with T2* modeling')
for i in range(max_num_iter):
    print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
    alg11.update()
print('')

recon_echo_1_w_decay_model = alg11.x

# remember that for the independent recons, we also use the 1st recon operator
alg12 = sigpy.alg.PrimalDualHybridGradient(
    proxfc=proxfc2,
    proxg=sigpy.prox.NoOp(A1.ishape),
    A=A1,
    AH=A1.H,
    x=deepcopy(recon_echo_2_wo_decay_model),
    u=u2,
    tau=1. / sigma,
    sigma=sigma,
    max_iter=max_num_iter)

print('recon echo 2- with T2* modeling')
for i in range(max_num_iter):
    print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
    alg12.update()
print('')

recon_echo_2_w_decay_model = alg12.x

del A1
del recon_operator_1
del recon_operator_2

cp.save(odir / 'recon_echo_1_w_decay_model.npy', recon_echo_1_w_decay_model)
cp.save(odir / 'recon_echo_2_w_decay_model.npy', recon_echo_2_w_decay_model)
#-------------------------------------------------------------------------
# calculate the ratio between the two recons without T2* decay modeling
# to estimate a monoexponential T2*

est_ratio_2 = cp.clip(
    cp.abs(recon_echo_2_w_decay_model) / cp.abs(recon_echo_1_w_decay_model), 0,
    1)
# set ratio to one in voxels where there is low signal in the first echo
mask = 1 - (cp.abs(recon_echo_1_w_decay_model) <
            0.05 * cp.abs(recon_echo_1_w_decay_model).max())

label, num_label = ndimage.label(mask == 1)
size = np.bincount(label.ravel())
biggest_label = size[1:].argmax() + 1
clump_mask = (label == biggest_label)

est_ratio_2[clump_mask == 0] = 1

cp.save(odir / 'est_ratio_2.npy', est_ratio_2)
#-------------------------------------------------------------------------
# setup the recon operators for the 1/2 echo using the estimated short T2* time (ratio)

recon_operator_1, recon_operator_2 = nufft_t2star_operator(
    ishape,
    kx,
    ky,
    kz,
    field_of_view_cm=field_of_view_cm,
    acq_sampling_time_ms=acq_sampling_time_ms,
    time_bin_width_ms=time_bin_width_ms,
    scale=scale,
    add_mirrored_coordinates=True,
    echo_time_1_ms=echo_time_1_ms,
    echo_time_2_ms=echo_time_2_ms,
    ratio_image=est_ratio_2)

A = sigpy.linop.Vstack([recon_operator_1, recon_operator_2, R])

proxfc = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_1.shape, 1, y=-data_echo_1),
    sigpy.prox.L2Reg(data_echo_2.shape, 1, y=-data_echo_2),
    sigpy.prox.Conj(proxg)
])

# setup the dual variable for both data forward models and the reg. operator
u3 = cp.concatenate(
    [u1[:data_echo_1.size], u2[:data_echo_2.size], u1[data_echo_1.size:]])

# norm of the two echo operator is slighly bigger -> smaller tau needed
alg3 = sigpy.alg.PrimalDualHybridGradient(
    proxfc=proxfc,
    proxg=sigpy.prox.NoOp(A.ishape),
    A=A,
    AH=A.H,
    x=deepcopy(recon_echo_1_w_decay_model),
    u=u3,
    tau=0.7 / sigma,
    sigma=sigma,
    max_iter=max_num_iter)

print('recon from both echos with T2* modeling')
for i in range(max_num_iter):
    print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
    alg3.update()
print('')

recon_both_echos = alg3.x

cp.save(odir / 'recon_both_echos.npy', recon_both_echos)

#---------------------------------------------------------------------------

r1 = np.abs(cp.asnumpy(recon_echo_1_w_decay_model))
r2 = np.abs(cp.asnumpy(recon_echo_2_w_decay_model))
rb = cp.asnumpy(recon_both_echos)

ims = 3 * [dict(vmin=0, vmax=3.5)] + [dict(vmin=0, vmax=1.)]

vi = pv.ThreeAxisViewer([
    np.flip(x, (0, 1)) for x in
    [np.abs(r1),
     np.abs(r2),
     np.abs(rb),
     np.abs(cp.asnumpy(est_ratio_2))]
],
                        imshow_kwargs=ims)
vi.fig.savefig(odir / 'recons.png', dpi=300)
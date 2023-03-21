#TODO save and check "rotation field" in rdb_hdr of pfile

import h5py
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndimage
import sigpy
import pymirc.viewer as pv
import nibabel as nib
import SimpleITK as sitk

from copy import deepcopy
from pathlib import Path

from utils import align_images, read_GE_ak_wav, kb_rolloff
from utils_sigpy import nufft_t2star_operator

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--beta_anatomical', type=float, default=1e-2)
parser.add_argument('--beta_non_anatomical', type=float, default=1e-1)
parser.add_argument('--max_num_iter', type=int, default=200)
args = parser.parse_args()

#--------------------------------------------------------------------
# input parameters
beta_anatomical = args.beta_anatomical
beta_non_anatomical = args.beta_non_anatomical
max_num_iter = args.max_num_iter

#--------------------------------------------------------------------
# fixed parameters
#gradient_file: str = '/data/sodium_mr/20230316_MR3_GS_QED/pfiles/g16/ak_grad56.wav'
#echo_1_data_file: str = '/data/sodium_mr/20230316_MR3_GS_QED/pfiles/g16/P30208.7.h5'
#echo_2_data_file: str = '/data/sodium_mr/20230316_MR3_GS_QED/pfiles/g16/P32256.7.h5'

gradient_file: str = '/data/sodium_mr/20230316_MR3_GS_QED/pfiles/g32/ak_grad59.wav'
echo_1_data_file: str = '/data/sodium_mr/20230316_MR3_GS_QED/pfiles/g32/P25600.7.h5'
echo_2_data_file: str = '/data/sodium_mr/20230316_MR3_GS_QED/pfiles/g32/P27648.7.h5'

t1_file: str = '/data/sodium_mr/20230316_MR3_GS_QED/niftis/t1.nii'

ishape = (128, 128, 128)

# regularization parameters for non-anatomy-guided recons
regularization_norm_non_anatomical = 'L2'

# regularization parameters for anatomy-guided recons
regularization_norm_anatomical = 'L1'

sigma = 0.1

time_bin_width_ms: float = 0.25

# echo times in ms
echo_time_1_ms = 0.455
echo_time_2_ms = 5.

odir = Path(echo_1_data_file).parent / 'recons'
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
field_of_view_cm = 100 * fov * 42.577 / gamma_by_2pi_MHz_T

#--------------------------------------------------------------------
#--------------------------------------------------------------------
# read the acquired data
#--------------------------------------------------------------------
#--------------------------------------------------------------------

with h5py.File(echo_1_data_file) as data1:
    data_echo_1 = data1['/data'][:]

with h5py.File(echo_2_data_file) as data2:
    data_echo_2 = data2['/data'][:]

# convert data to complex (from two reals) send data arrays to GPU
data_echo_1 = cp.asarray(data_echo_1['real'] + 1j * data_echo_1['imag'])
data_echo_2 = cp.asarray(data_echo_2['real'] + 1j * data_echo_2['imag'])

# scale the data such that we get CSF approx 3 with normalized nufft operator
data_scale = 3. / cp.abs(data_echo_1)[0, :].mean()

data_echo_1 *= data_scale
data_echo_2 *= data_scale

# ignore the first time point in the readout
data_echo_1 = data_echo_1[1:, :]
data_echo_2 = data_echo_2[1:, :]
k_1_cm = k_1_cm[1:, :]

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
#--------------------------------------------------------------------
#--------------------------------------------------------------------
# regrid the data and do simple IFFT
#--------------------------------------------------------------------
#--------------------------------------------------------------------

# flatten the data for the nufft operators
data_echo_1 = data_echo_1.ravel()
data_echo_2 = data_echo_2.ravel()

# grid data for conventional IFFT recon
kernel = 'kaiser_bessel'
width = 2
param = 9.14
grid_shape = ishape

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

# setup a phase correction field to account phase definition in numpy's fft
tmp_x = np.arange(grid_shape[0])
TMP_X, TMP_Y, TMP_Z = np.meshgrid(tmp_x, tmp_x, tmp_x)

phase_corr = cp.asarray(((-1)**TMP_X) * ((-1)**TMP_Y) * ((-1)**TMP_Z))

ifft_scale = 50.

ifft1 = ifft_scale * phase_corr * ifft_op(data_echo_1_gridded_corr)
ifft2 = ifft_scale * phase_corr * ifft_op(data_echo_2_gridded_corr)

tmp_x = cp.linspace(-width / 2, width / 2, grid_shape[0])
TMP_X, TMP_Y, TMP_Z = cp.meshgrid(tmp_x, tmp_x, tmp_x)
R = cp.sqrt(TMP_X**2 + TMP_Y**2 + TMP_Z**2)
R = cp.clip(R, 0, tmp_x.max())
#TODO: understand why factor 1.6 is needed when regridding in 3D
interpolation_correction_field = kb_rolloff(1.6 * R, param)
interpolation_correction_field /= interpolation_correction_field.max()

ifft1 /= interpolation_correction_field
ifft2 /= interpolation_correction_field

ifft1_sm = ndimage.gaussian_filter(ifft1, 1.5)
ifft2_sm = ndimage.gaussian_filter(ifft2, 1.5)

cp.save(odir / 'ifft1.npy', ifft1)
cp.save(odir / 'ifft2.npy', ifft2)
cp.save(odir / 'ifft1_sm.npy', ifft1_sm)
cp.save(odir / 'ifft2_sm.npy', ifft2_sm)

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# setup the operators for reconstruction and regularization

# scale is needed to get normalized operator
nufft_single_echo_no_decay = nufft_t2star_operator(
    ishape,
    k_1_cm,
    field_of_view_cm=field_of_view_cm,
    acq_sampling_time_ms=acq_sampling_time_ms,
    time_bin_width_ms=time_bin_width_ms,
    scale=1.,
    add_mirrored_coordinates=False)

max_eig_nufft_single = sigpy.app.MaxEig(nufft_single_echo_no_decay.H *
                                        nufft_single_echo_no_decay,
                                        dtype=cp.complex128,
                                        device=data_echo_1.device,
                                        max_iter=30).run()

nufft_single_echo_no_decay = (
    1 / np.sqrt(max_eig_nufft_single)) * nufft_single_echo_no_decay

# set up the operator for regularization
G = (1 / np.sqrt(12)) * sigpy.linop.FiniteDifference(ishape, axes=None)
#------------------------------------------------------
# reconstruct the first echo without T2* decay modeling

A = sigpy.linop.Vstack([nufft_single_echo_no_decay, G])

# estimate the norm of the stacked nufft and gradient operator
# which we need for the step sizes of the PDHG
max_eig = sigpy.app.MaxEig(A.H * A,
                           dtype=cp.complex128,
                           device=data_echo_1.device,
                           max_iter=30).run()

if regularization_norm_non_anatomical == 'L2':
    proxg = sigpy.prox.L2Reg(G.oshape, lamda=beta_non_anatomical)
elif regularization_norm_non_anatomical == 'L1':
    proxg = sigpy.prox.L1Reg(G.oshape, lamda=beta_non_anatomical)
else:
    raise ValueError('unknown regularization norm')

# estimate norm of the nufft operator if not given
proxfc1 = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_1.shape, 1, y=-data_echo_1),
    sigpy.prox.Conj(proxg)
])
u1 = cp.zeros(A.oshape, dtype=data_echo_1.dtype)

outfile1 = odir / f'recon_echo_1_no_decay_model_{regularization_norm_non_anatomical}_{beta_non_anatomical:.1E}_{max_num_iter}.npz'

if not outfile1.exists():
    alg1 = sigpy.alg.PrimalDualHybridGradient(proxfc=proxfc1,
                                              proxg=sigpy.prox.NoOp(A.ishape),
                                              A=A,
                                              AH=A.H,
                                              x=deepcopy(ifft1),
                                              u=u1,
                                              tau=1 / (max_eig * sigma),
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

#-----------------------------------------------------
proxfc2 = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_1.shape, 1, y=-data_echo_2),
    sigpy.prox.Conj(proxg)
])
u2 = cp.zeros(A.oshape, dtype=data_echo_2.dtype)

outfile2 = odir / f'recon_echo_2_no_decay_model_{regularization_norm_non_anatomical}_{beta_non_anatomical:.1E}_{max_num_iter}.npz'

if not outfile2.exists():
    alg2 = sigpy.alg.PrimalDualHybridGradient(proxfc=proxfc2,
                                              proxg=sigpy.prox.NoOp(A.ishape),
                                              A=A,
                                              AH=A.H,
                                              x=deepcopy(ifft2),
                                              u=u2,
                                              tau=1. / (max_eig * sigma),
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

#---------------------------------------------------------------------
#---------------------------------------------------------------------
# load and align the 1H MR image
#---------------------------------------------------------------------
#---------------------------------------------------------------------

t1_nii = nib.load(t1_file)
t1_nii = nib.as_closest_canonical(t1_nii)
t1 = t1_nii.get_fdata()

t1_affine = t1_nii.affine
t1_voxsize = t1_nii.header['pixdim'][1:4]
t1_origin = t1_affine[:-1, -1]

na_voxsize = 10 * field_of_view_cm / np.array(ishape)
na_origin = t1_origin.copy()

# save the Na origin such that we can later transform the
# Na recons back to other grids
np.savetxt(odir / 'na_origin.txt', na_origin)

t1_aligned_file = odir / 't1_aligned.npy'

if not t1_aligned_file.exists():
    print('aligning the 1H MR image')
    t1_aligned, final_transform = align_images(
        np.abs(cp.asnumpy(recon_echo_1_wo_decay_model)), t1, na_voxsize,
        na_origin, t1_voxsize, t1_origin)

    # send aligned image to GPU
    t1_aligned = cp.asarray(t1_aligned)
    cp.save(t1_aligned_file, t1_aligned)
    sitk.WriteTransform(final_transform, str(odir / 't1_transform.tfm'))
else:
    t1_aligned = cp.load(t1_aligned_file)

# normalize the intensity of the aligned t1 image (not needed, just for convenience)
t1_aligned /= cp.percentile(t1_aligned, 99.9)

#---------------------------------------------------------------------

ims = 2 * [dict(vmin=0, vmax=3.5, cmap='Greys_r')] + [dict(cmap='Greys_r')]
vi = pv.ThreeAxisViewer([
    np.abs(cp.asnumpy(recon_echo_1_wo_decay_model)),
    np.abs(cp.asnumpy(recon_echo_2_wo_decay_model)),
    cp.asnumpy(t1_aligned)
], [None, None, np.abs(cp.asnumpy(recon_echo_1_wo_decay_model))],
                        imshow_kwargs=ims)

#---------------------------------------------------------------------
# projected gradient operator that we need for DTV
xi = G(t1_aligned)

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

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

A = sigpy.linop.Vstack([nufft_single_echo_no_decay, PG])

if regularization_norm_anatomical == 'L2':
    proxg = sigpy.prox.L2Reg(G.oshape, lamda=beta_anatomical)
elif regularization_norm_anatomical == 'L1':
    proxg = sigpy.prox.L1Reg(G.oshape, lamda=beta_anatomical)
else:
    raise ValueError('unknown regularization norm')

proxfc1 = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_1.shape, 1, y=-data_echo_1),
    sigpy.prox.Conj(proxg)
])

outfile1 = odir / f'agr_echo_1_no_decay_model_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{max_num_iter}.npz'

if not outfile1.exists():
    alg1 = sigpy.alg.PrimalDualHybridGradient(
        proxfc=proxfc1,
        proxg=sigpy.prox.NoOp(A.ishape),
        A=A,
        AH=A.H,
        x=deepcopy(recon_echo_1_wo_decay_model),
        u=u1,
        tau=1. / sigma,
        sigma=sigma,
        max_iter=max_num_iter)

    print('AGR echo 1 - no T2* modeling')
    for i in range(max_num_iter):
        print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
        alg1.update()
    print('')

    cp.savez(outfile1, x=alg1.x, u=alg1.u)
    agr_echo_1_wo_decay_model = alg1.x
else:
    d1 = cp.load(outfile1)
    agr_echo_1_wo_decay_model = d1['x']
    u1 = d1['u']

#----------------------------------------------------------------------
# AGR of 2nd echo without decay modeling

proxfc2 = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_2.shape, 1, y=-data_echo_2),
    sigpy.prox.Conj(proxg)
])

outfile2 = odir / f'agr_echo_2_no_decay_model_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{max_num_iter}.npz'

if not outfile2.exists():
    alg2 = sigpy.alg.PrimalDualHybridGradient(
        proxfc=proxfc2,
        proxg=sigpy.prox.NoOp(A.ishape),
        A=A,
        AH=A.H,
        x=deepcopy(recon_echo_2_wo_decay_model),
        u=u2,
        tau=1. / sigma,
        sigma=sigma,
        max_iter=max_num_iter)

    print('AGR echo 2 - no T2* modeling')
    for i in range(max_num_iter):
        print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
        alg2.update()
    print('')

    cp.savez(outfile2, x=alg2.x, u=alg2.u)
    agr_echo_2_wo_decay_model = alg2.x
else:
    d2 = cp.load(outfile2)
    agr_echo_2_wo_decay_model = d2['x']
    u2 = d2['u']

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

##-----------------------------------------------------------------------------
##-----------------------------------------------------------------------------
## perform independent AGRs with decay modeling
##-----------------------------------------------------------------------------
##-----------------------------------------------------------------------------
##
#recon_operator_1, recon_operator_2 = nufft_t2star_operator(
#    ishape,
#    k_1_cm,
#    field_of_view_cm=field_of_view_cm,
#    acq_sampling_time_ms=acq_sampling_time_ms,
#    time_bin_width_ms=time_bin_width_ms,
#    scale=1 / np.sqrt(max_eig_nufft_single),
#    add_mirrored_coordinates=False,
#    echo_time_1_ms=echo_time_1_ms,
#    echo_time_2_ms=echo_time_2_ms,
#    ratio_image=est_ratio)
#
#A = sigpy.linop.Vstack([recon_operator_1, PG])
#
#if regularization_norm_anatomical == 'L2':
#    proxg = sigpy.prox.L2Reg(PG.oshape, lamda=beta_anatomical)
#elif regularization_norm_anatomical == 'L1':
#    proxg = sigpy.prox.L1Reg(PG.oshape, lamda=beta_anatomical)
#else:
#    raise ValueError('unknown regularization norm')
#
#proxfc1 = sigpy.prox.Stack([
#    sigpy.prox.L2Reg(data_echo_1.shape, 1, y=-data_echo_1),
#    sigpy.prox.Conj(proxg)
#])
#
#outfile1 = odir / f'agr_echo_1_with_decay_model_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{max_num_iter}.npz'
#
#if not outfile1.exists():
#    alg1 = sigpy.alg.PrimalDualHybridGradient(
#        proxfc=proxfc1,
#        proxg=sigpy.prox.NoOp(A.ishape),
#        A=A,
#        AH=A.H,
#        x=deepcopy(agr_echo_1_wo_decay_model),
#        u=u1,
#        tau=1 / sigma,
#        sigma=sigma,
#        max_iter=max_num_iter)
#
#    print('AGR echo 1 - with T2* modeling')
#    for i in range(max_num_iter):
#        print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
#        alg1.update()
#    print('')
#
#    cp.savez(outfile1, x=alg1.x, u=alg1.u)
#    agr_echo_1_w_decay_model = alg1.x
#else:
#    d1 = cp.load(outfile1)
#    agr_echo_1_w_decay_model = d1['x']
#    u1 = d1['u']
#
##----------------------------------------------------------------------
## AGR of 2nd echo with decay modeling
#
#proxfc2 = sigpy.prox.Stack([
#    sigpy.prox.L2Reg(data_echo_2.shape, 1, y=-data_echo_2),
#    sigpy.prox.Conj(proxg)
#])
#
#outfile2 = odir / f'agr_echo_2_with_decay_model_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{max_num_iter}.npz'
#
#if not outfile2.exists():
#    alg2 = sigpy.alg.PrimalDualHybridGradient(
#        proxfc=proxfc2,
#        proxg=sigpy.prox.NoOp(A.ishape),
#        A=A,
#        AH=A.H,
#        x=deepcopy(recon_echo_2_wo_decay_model),
#        u=u2,
#        tau=1. / sigma,
#        sigma=sigma,
#        max_iter=max_num_iter)
#
#    print('AGR echo 2 - with T2* modeling')
#    for i in range(max_num_iter):
#        print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
#        alg2.update()
#    print('')
#
#    cp.savez(outfile2, x=alg2.x, u=alg2.u)
#    agr_echo_2_w_decay_model = alg2.x
#else:
#    d2 = cp.load(outfile2)
#    agr_echo_2_w_decay_model = d2['x']
#    u2 = d2['u']
#
#del A
#
##---------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------
## recons with "estimated" decay model and anatomical prior using data from both echos
##---------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------
#
## the two echos usually have different phases
## we correct for this by multiplying by the negative estimated phases
#phase_fac_1 = cp.exp(1j * cp.angle(agr_echo_1_w_decay_model))
#phase_fac_2 = cp.exp(1j * cp.angle(agr_echo_2_w_decay_model))
#
#A = sigpy.linop.Vstack([
#    recon_operator_1 * sigpy.linop.Multiply(ishape, phase_fac_1),
#    recon_operator_2 * sigpy.linop.Multiply(ishape, phase_fac_2), PG
#])
#
#proxfcb = sigpy.prox.Stack([
#    sigpy.prox.L2Reg(data_echo_1.shape, 1, y=-data_echo_1),
#    sigpy.prox.L2Reg(data_echo_2.shape, 1, y=-data_echo_2),
#    sigpy.prox.Conj(proxg)
#])
#
#ub = cp.zeros(A.oshape, dtype=cp.complex128)
#
#outfileb = odir / f'agr_both_echo_w_decay_model_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{max_num_iter}.npz'
#
#if not outfileb.exists():
#    algb = sigpy.alg.PrimalDualHybridGradient(
#        proxfc=proxfcb,
#        proxg=sigpy.prox.NoOp(A.ishape),
#        A=A,
#        AH=A.H,
#        x=deepcopy(agr_echo_1_w_decay_model),
#        u=ub,
#        tau=0.5 / sigma,
#        sigma=sigma,
#        max_iter=max_num_iter)
#
#    print('AGR both echos - "estimated" T2* modeling')
#    for i in range(max_num_iter):
#        print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
#        algb.update()
#    print('')
#
#    cp.savez(outfileb, x=algb.x, u=ub)
#    agr_both_echos_w_decay_model = algb.x
#else:
#    db = cp.load(outfileb)
#    agr_both_echos_w_decay_model = db['x']
#    ub = db['u']
#
#del A
#del recon_operator_1
#del recon_operator_2
#
##-----------------------------------------------------------------------------
#
#ims = 4 * [dict(vmin=0, vmax=4., cmap='Greys_r')] + [dict(cmap='Greys_r')]
#vi = pv.ThreeAxisViewer([
#    np.abs(cp.asnumpy(recon_echo_1_wo_decay_model)),
#    np.abs(cp.asnumpy(agr_echo_1_wo_decay_model)),
#    np.abs(cp.asnumpy(agr_echo_1_w_decay_model)),
#    np.abs(cp.asnumpy(agr_both_echos_w_decay_model)),
#    cp.asnumpy(est_ratio)
#],
#                        imshow_kwargs=ims)
#
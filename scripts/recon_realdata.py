""" Reconstruct real raw dual echo TPI Sodium data

    Fixed compartmental T2* model:
    - knowns: short and long T2* components and their ratio for a biexponential compartment, long T2* component for a monoexponential compartment (fluid)
    - unknowns: "concentrations" for each compartment

    Monoexponential T2* model:
    - unknowns: monoexponential T2* map (Gamma = T2* decay between TE1 and TE2), "concentration"
"""

import warnings

try:
    import cupy as cp
except ImportError:
    warnings.warn("cupy package not available", RuntimeWarning)
    

import numpy as np
import nibabel as nib

from scipy.ndimage     import zoom, gaussian_filter
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.optimize import fmin_l_bfgs_b

import matplotlib.pyplot as plt

from copy import deepcopy
from argparse import ArgumentParser
import sys
import os
from glob import glob
from datetime import datetime

import pymirc.viewer as pv
import pynamr

parser = ArgumentParser(description = '3D real Na MRI dual echo reconstruction')
parser.add_argument('case')
parser.add_argument('--pathology',  default = 'TBI', choices = ['TBI','CSF'])
parser.add_argument('--sdir',  default = '/uz/data/Admin/ngeworkingresearch/MarinaFilipovic/SodiumMRIdata/')
parser.add_argument('--niter',  default = 20, type = int, help='number of optimization iterations')
parser.add_argument('--n_outer', default = 10, type = int, help='number of outer optimization iterations if multistep')
parser.add_argument('--beta_x', default = 1e-2, type = float, help='penalty strength for image/param')
parser.add_argument('--beta_gam', default = 1e-2, type = float, help='penalty strength for T2* decay')
parser.add_argument('--n', default = 128, type = int, choices = [128,256], help='image size')
parser.add_argument('--n_readout_bins', default = 16, type = int, help='TPI readout bins')
parser.add_argument('--nnearest', default = 13,  type = int, help='Bowsher number of most similar voxels')
parser.add_argument('--instant_tpi_recon',   default = False, action='store_true', help='TPI readout instantaneous for reconstruction')
parser.add_argument('--model_recon',   default = 'monoexp', type = str, choices = ['monoexp','fixedcomp','custom'],
                       help='forward model for reconstructing the raw data')
parser.add_argument('--t2bi_s', default = 3., type = float, help='fixed T2* biexponential short component')
parser.add_argument('--t2bi_l', default = 20., type = float, help='fixed T2* biexponential long component')
parser.add_argument('--t2mono_l', default = 25., type = float, help='fixed T2* monoexponential component (i.e. fluid)')
parser.add_argument('--t2bi_frac_l', default = 0.4, type = float, help='fixed T2* biexponential fraction of long component')
parser.add_argument('--t2bi_smap_filename', default = None, type = str, help='T2* biexponential short component spatial map')
parser.add_argument('--t2bi_lmap_filename', default = None, type = str, help='T2* biexponential long component spatial map')
parser.add_argument('--t2mono_map_filename', default = None, type = str, help='T2* monoexponential (long) component spatial map (i.e. fluid)')
parser.add_argument('--highres_mri_name', default = 'mprage', help='higher resolution mri name')
parser.add_argument('--no_highres_prior', action = 'store_true', help='no additional higher resolution mri')
parser.add_argument('--load_results', action = 'store_true', help='load already computed results, display and exit')


args = parser.parse_args()

niter       = args.niter
n_outer     = args.n_outer
beta_x      = args.beta_x
beta_gam    = args.beta_gam
n           = args.n
n_readout_bins     = args.n_readout_bins
nnearest    = args.nnearest
instant_tpi_recon = args.instant_tpi_recon
model_recon = args.model_recon
t2bi_s      = args.t2bi_s
t2bi_l      = args.t2bi_l
t2bi_frac_l = args.t2bi_frac_l
t2mono_l    = args.t2mono_l
t2bi_smap_filename = args.t2bi_smap_filename
t2bi_lmap_filename = args.t2bi_lmap_filename
t2mono_map_filename = args.t2mono_map_filename
highres_mri_name = args.highres_mri_name
no_highres_prior = args.no_highres_prior
sdir        = args.sdir
case        = args.case
pathology  = args.pathology
load_results = args.load_results


#-------------------------------------------------------------------------------------
# Utility functions for results
#-------------------------------------------------------------------------------------

# display reconstruction results
def display_results():
    if model_recon=="monoexp":
        ims_1 = dict(cmap=plt.cm.viridis, vmin = 0, vmax = std_te1.max())
        ims_2 = dict(cmap=plt.cm.viridis, vmin = 0, vmax = 1.)
        t1_1 = dict(cmap=plt.cm.gray, vmin = 0, vmax = highres_proton_img.max())

        # empirical rough estimate of gamma for TBI data
        gam_rough_est = np.divide(std_te2, std_te1, where=std_te1>0.1*std_te1.max())
        gam_rough_est[std_te1<=0.1*std_te1.max()]=1.

        # reconstructed image at time 0 and simple recon at TE1
        vi_x_te1 = pv.ThreeAxisViewer([std_te1,
                         np.linalg.norm(x_r, axis=-1),
                         highres_proton_img,
                         gam_r],
                         imshow_kwargs=[ims_1, ims_1, t1_1, ims_2])

        # reconstructed Gamma and a rough estimation from simple recons
        vi_gam = pv.ThreeAxisViewer([gam_rough_est,
                         gam_r],
                         imshow_kwargs=[ims_2, ims_2])

        # reconstructed and simple TE1 and TE2 images
        vi_te1_te2 = pv.ThreeAxisViewer([std_te1,
                         np.linalg.norm(x_r, axis=-1)*gam_r**(te1/delta_t),
                         std_te2,
                         np.linalg.norm(x_r, axis=-1)*gam_r**(1+(te1/delta_t))],
                         imshow_kwargs=[ims_1, ims_1, ims_1, ims_1])
    elif model_recon=="fixedcomp":
        ims_1 = dict(cmap=plt.cm.viridis, vmin = 0, vmax = std_te1.max())

        vi = pv.ThreeAxisViewer([np.linalg.norm(x_r[0], axis=-1),
                         np.linalg.norm(x_r[1], axis=-1),
                         std_te1_filtered],
                         imshow_kwargs=[ims_1, ims_1, ims_1])
    else:
        raise NotImplementedError

# save reconstruction results to files
def save_results():
    # write input arguments to file
    with open(os.path.join(odir,'input_params.csv'), 'w') as f:
        for x in args.__dict__.items():
            f.write("%s,%s\n"%(x[0],x[1]))
    # write images
    if model_recon=="monoexp":
        x_r.tofile(os.path.join(odir,'x_r.img'))
        gam_r.tofile(os.path.join(odir,'gam_r.img'))
    elif model_recon=="fixedcomp":
        x_r.tofile(os.path.join(odir,'x_r.img'))
    else:
        raise NotImplementedError


#-------------------------------------------------------------------------------------
# Load and preprocess "raw" data
#-------------------------------------------------------------------------------------

# input folders and specific parameters
if pathology=='TBI':
    casedir = f"TBI-0{args.case}"
    sdir1 = os.path.join(args.sdir, casedir, 'TE03', 'DeNoise')
    sdir2 = os.path.join(args.sdir, casedir, 'TE5', 'DeNoise')
    fpattern = '*.c?'
    ncoils = 8
    data_n = 64
    te1 = 0.3
    delta_t = 4.7
elif pathology=='CSF':
    casedir = f"CSF_H0{args.case}b1_birdcage"
    sdir1 = os.path.join(args.sdir, casedir, 'TE05','DeNoise')
    sdir2 = os.path.join(args.sdir, casedir, 'TE5', 'DeNoise')
    fpattern = '*.c?'
    ncoils  = 1
    data_n = 64
    te1 = 0.5
    delta_t = 4.5

# find files
fnames1 = sorted(glob(os.path.join(sdir1, fpattern)))
fnames2 = sorted(glob(os.path.join(sdir2, fpattern)))
if (len(fnames1) != ncoils) or (len(fnames2) != ncoils):
    raise ValueError('The number of data files is not correct')
odir = os.path.join(sdir, casedir, f'betax_{beta_x:.1E}'+ (f'_betagam_{beta_gam:.1E}' if model_recon=='monoexp' else '')+
                                   (f'_{highres_mri_name}_nbow_{nnearest}' if not no_highres_prior else ''))
if not os.path.exists(odir):
  os.makedirs(odir)

# high resolution registered proton MRI image
if not no_highres_prior:
    highres_proton_img_file = os.path.join(sdir, casedir, f"proton/{highres_mri_name}_n4_aligned_128.npy")


# initialize some parameters and variables
# 2 TE values currently
nechos = 2
# shape of the kspace data of a single coil
data_shape = (data_n, data_n, data_n)
# shape of the reconstructed image
recon_shape = (n, n, n)
# down sample factor (recon cube size / data cube size), must be integer
ds = n//data_n
# initialize variables
data           = np.zeros((ncoils, nechos) + data_shape,  dtype = np.complex64)
data_pad       = np.zeros((ncoils, nechos) + recon_shape,  dtype = np.complex64)
sens           = np.zeros((ncoils,) + recon_shape,  dtype = np.complex64)

# calculate gaussian filter in k-space used for sensitivity computations
# matches a sigma=8vox in image space
k = np.fft.fftfreq(recon_shape[0])
k0,k1,k2 = np.meshgrid(k, k, k, indexing = 'ij')
abs_k    = np.sqrt(k0**2 + k1**2 + k2**2)
filt     =  np.exp(-(abs_k**2)/(2*0.02**2))

# factor for compensating the difference in fft normalization factors
# when going back and forth between ffts with different N, given norm='ortho'
# for padded data
factor_fft_diff_res = np.sqrt(ds**3)

# load data
for i in range(ncoils):
  # load image space data of first echo
  cimg = np.flip(np.fromfile(fnames1[i], dtype = np.complex64).reshape(data_shape).swapaxes(0,2), (1,2))

  # perform fft to get into k-space
  data[i,0] = np.fft.fftn(cimg, norm = 'ortho')
  # pad the data to recon resolution
  data_pad[i,0] = np.fft.fftshift(np.pad(np.fft.fftshift(data[i,0]), (recon_shape[0] - data_shape[0])//2))
  # compensate the difference in fft normalization factors for data and padded data resolutions
  data_pad[i,0] *= factor_fft_diff_res

  # load image space data of 2nd echo
  cimg = np.flip(np.fromfile(fnames2[i], dtype = np.complex64).reshape(data_shape).swapaxes(0,2), (1,2))
  data[i,1]    = np.fft.fftn(cimg, norm = 'ortho')
  # pad the data to recon resolution
  data_pad[i,1] = np.fft.fftshift(np.pad(np.fft.fftshift(data[i,1]), (recon_shape[0] - data_shape[0])//2))
  # compensate the difference in fft normalization factors for data and padded data resolutions
  data_pad[i,1] *= factor_fft_diff_res 

# calculate the sum of square image
sos_filt_te1 = pynamr.sum_of_squares_reconstruction(data_pad[:,0]*filt, complex_format=True)

# compute sensitivities
for i in range(ncoils):
    coil_im = pynamr.simple_reconstruction(data_pad[i,0]*filt, complex_format=True)
    sens[i,...]   = coil_im / sos_filt_te1

# scale signal to be not too far away from 1
scale_factor = np.max(sos_filt_te1)
data = data / scale_factor

# higher res proton image
if not no_highres_prior:
    highres_proton_img = np.load(highres_proton_img_file)


#-------------------------------------------------------------------------------------
# "Standard/simple" recon
#-------------------------------------------------------------------------------------

# currently sum of squares, TODO implement a better std recon
std_te1 = pynamr.sum_of_squares_reconstruction(data_pad[:,0,...], complex_format=True)
std_te2 = pynamr.sum_of_squares_reconstruction(data_pad[:,1,...], complex_format=True)

# filter the SOS images
std_te1_filtered = gaussian_filter(std_te1, 1.)
std_te2_filtered = gaussian_filter(std_te2, 1.)


#-------------------------------------------------------------------------------------
# Load and display already computed reconstruction results and exit
#-------------------------------------------------------------------------------------

if load_results:
    if model_recon == "monoexp":
        x_r = np.fromfile(os.path.join(odir,'x_r.img'), dtype=np.float64).reshape(recon_shape+(2,))
        gam_r = np.fromfile(os.path.join(odir,'gam_r.img'), dtype=np.float64).reshape(recon_shape)
        display_results()
    elif model_recon == "fixedcomp":
        x_r = np.fromfile(os.path.join(odir,'x_r.img'), dtype=np.float64).reshape((2,)+recon_shape+(2,))
        display_results()
    sys.exit()


#-------------------------------------------------------------------------------------
# Reconstruction
#-------------------------------------------------------------------------------------

# convert variables to suitable formats
data = pynamr.real_view_of_complex_array(data)

# k-space config
kspace_part = pynamr.RadialKSpacePartitioner(data_shape, n_readout_bins)

# readout config
if instant_tpi_recon:
    readout_time = pynamr.TPIInstantaneousReadOutTime()
else:
    readout_time = pynamr.TPIReadOutTime()

# forward model and unknowns for reconstruction
if model_recon == "monoexp":
    fwd_model = pynamr.MonoExpDualTESodiumAcqModel(ds, sens, delta_t, te1, readout_time, kspace_part)
    unknowns = {pynamr.VarName.IMAGE: pynamr.Var(shape=tuple([ds * x for x in data_shape]) + (2,)),
                    pynamr.VarName.GAMMA: pynamr.Var(shape=tuple([ds * x for x in data_shape]), nb_comp=1, complex_var=False)}
elif model_recon == "fixedcomp":
    fwd_model = pynamr.TwoCompartmentBiExpDualTESodiumAcqModel(ds, sens, delta_t, te1, readout_time, kspace_part, 0, t2mono_l, t2bi_s, t2bi_l, 1, t2bi_frac_l)
    unknowns = {pynamr.VarName.PARAM: pynamr.Var(shape=tuple([2,] + [ds * x for x in data_shape] + [2,]), nb_comp=2)}
else:
    raise NotImplementedError

#-------------------------------------------------------------------------------------
# setup the data fidelity loss function
data_fidelity_loss = pynamr.DataFidelityLoss(fwd_model, data)

#-------------------------------------------------------------------------------------
# setup the priors
# simulate a perfect anatomical prior image (with changed contrast but matching edges)
aimg = highres_proton_img.copy()
bowsher_loss = pynamr.generate_bowsher_loss(aimg, nnearest)

#-------------------------------------------------------------------------------------
# setup the total loss function consiting of data fidelity loss and the priors
if model_recon == "monoexp":
    penalty_info = {pynamr.VarName.IMAGE: bowsher_loss, pynamr.VarName.GAMMA: bowsher_loss}
    beta_info = {pynamr.VarName.IMAGE: beta_x, pynamr.VarName.GAMMA: beta_gam}
elif model_recon == "fixedcomp":
    penalty_info = {pynamr.VarName.PARAM: bowsher_loss}
    beta_info = {pynamr.VarName.PARAM: beta_x}
else:
    raise NotImplementedError

loss = pynamr.TotalLoss(data_fidelity_loss, penalty_info, beta_info)


#-------------------------------------------------------------------------------------
# load existing results


#-------------------------------------------------------------------------------------
# run the recons
if model_recon=="monoexp":
    # allocate initial values of unknown variables
    unknowns[pynamr.VarName.IMAGE].value = np.stack([std_te1_filtered, 0*std_te1_filtered], axis=-1)
    unknowns[pynamr.VarName.GAMMA].value = np.clip(std_te2_filtered / (std_te1_filtered + 1e-7), 0, 1)

    #-------------------------------------------------------------------------------------
    # alternating LBFGS steps
    for i_out in range(n_outer):

        var_name = pynamr.VarName.IMAGE
        # update complex sodium image
        res_1 = fmin_l_bfgs_b(loss,
                             (unknowns[var_name].value).copy().ravel(),
                             fprime=loss.gradient,
                             args=(deepcopy(unknowns), var_name),
                             maxiter=niter,
                             disp=1)

        # update current value
        unknowns[pynamr.VarName.IMAGE].value = res_1[0].copy().reshape(unknowns[pynamr.VarName.IMAGE].shape)

        var_name = pynamr.VarName.GAMMA
        # update real gamma (decay) image
        res_2 = fmin_l_bfgs_b(loss,
                              (unknowns[var_name].value).copy().ravel(),
                              fprime=loss.gradient,
                              args=(deepcopy(unknowns), var_name),
                              maxiter=niter,
                              disp=1,
                              bounds=(unknowns[var_name].value.size) * [(0.001, 1)])

        # update current value
        unknowns[pynamr.VarName.GAMMA].value = res_2[0].copy().reshape(unknowns[pynamr.VarName.GAMMA].shape)

    # postprocess results
    x_r = unknowns[pynamr.VarName.IMAGE].value * scale_factor
    gam_r = unknowns[pynamr.VarName.GAMMA].value

    # save the results
    save_results()

    # show the results
    display_results()

elif model_recon=="fixedcomp":
    # allocate arrays for recons and copy over initial values
    unknowns[pynamr.VarName.PARAM].value = np.ones(unknowns[pynamr.VarName.PARAM].shape, np.float64)

    # run the recon
    res_1 = fmin_l_bfgs_b(loss,
                          unknowns[pynamr.VarName.PARAM].value.copy().ravel(),
                          fprime=loss.gradient,
                          args=(unknowns, pynamr.VarName.PARAM),
                          maxiter=niter,
                          disp=1)

    # postprocess the results
    unknowns[pynamr.VarName.PARAM].value = res_1[0].copy().reshape(unknowns[pynamr.VarName.PARAM].shape)
    x_r = unknowns[pynamr.VarName.PARAM].value * scale_factor

    # save the results
    save_results()

    # show the results
    display_results()

else:
    raise NotImplementedError



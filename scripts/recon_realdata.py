""" Reconstruct real raw dual echo TPI Sodium data

    Fixed compartmental T2* model:
    - knowns: short and long T2* components and their ratio for a biexponential compartment, long T2* component for a monoexponential compartment (fluid)
    - unknowns: "concentrations" for each compartment

    Monoexponential T2* model:
    - unknowns: monoexponential T2* map (Gamma = T2* decay between TE1 and TE2), "concentration"

    Data NYU: require some fast preprocessing, only the preprocessed higher resolution H MRI image is saved, other things are computed on the fly 

    TODO:
    - find a way to proceed to recon after checking the preprocessed data with pymirc viewer, instead of exiting and reexecuting the script
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
parser.add_argument('--t2bi_frac_l', default = 0.6, type = float, help='fixed T2* biexponential fraction of long component')
parser.add_argument('--t2bi_smap_filename', default = None, type = str, help='T2* biexponential short component spatial map')
parser.add_argument('--t2bi_lmap_filename', default = None, type = str, help='T2* biexponential long component spatial map')
parser.add_argument('--t2mono_map_filename', default = None, type = str, help='T2* monoexponential (long) component spatial map (i.e. fluid)')
parser.add_argument('--highres_mri_name', default = 'mprage', help='higher resolution mri name')
parser.add_argument('--no_highres_prior', action = 'store_true', help='no additional higher resolution mri')
parser.add_argument('--load_results', action = 'store_true', help='load already computed results, display and exit')
parser.add_argument('--check', action = 'store_true', help='check the preprocessed data and exit')


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
check      = args.check

#-------------------------------------------------------------------------------------
# Utility functions for results
#-------------------------------------------------------------------------------------

# display reconstruction results
def display_results():
    if model_recon=="monoexp":
        ims_1 = dict(cmap=plt.cm.viridis, vmin = 0, vmax = std_te1.max())
        ims_2 = dict(cmap=plt.cm.viridis, vmin = 0, vmax = 1.)
        highresH_img_1 = dict(cmap=plt.cm.gray, vmin = 0, vmax = highresH_img.max())

        # empirical rough estimate of gamma for TBI data
        gam_rough_est = np.divide(std_te2, std_te1, where=std_te1>0.1*std_te1.max())
        gam_rough_est[std_te1<=0.1*std_te1.max()]=1.

        # reconstructed image at time 0 and simple recon at TE1
        vi_x_te1 = pv.ThreeAxisViewer([std_te1,
                         np.abs(x_r),
                         highresH_img,
                         gam_r],
                         imshow_kwargs=[ims_1, ims_1, highresH_img_1, ims_2])

        # reconstructed Gamma and a rough estimation from simple recons
        vi_gam = pv.ThreeAxisViewer([gam_rough_est,
                         gam_r],
                         imshow_kwargs=[ims_2, ims_2])

        # reconstructed and simple TE1 and TE2 images
        vi_te1_te2 = pv.ThreeAxisViewer([std_te1,
                         np.abs(x_r)*gam_r**(te1/delta_t),
                         std_te2,
                         np.abs(x_r)*gam_r**(1+(te1/delta_t))],
                         imshow_kwargs=[ims_1, ims_1, ims_1, ims_1])
        return (vi_x_te1, vi_gam, vi_te1_te2)
    elif model_recon=="fixedcomp":
        ims_1 = dict(cmap=plt.cm.viridis, vmin = 0, vmax = std_te1.max())

        vi = pv.ThreeAxisViewer([np.abs(x_r[0]),
                                 np.abs(x_r[1]),
                                 std_te1_filtered],
                                 imshow_kwargs=[ims_1, ims_1, ims_1])
        return (vi,)
    else:
        raise NotImplementedError
    print("Displayed reconstruction results")

# save reconstruction results to files
def save_results():
    if not os.path.exists(odir):
        os.makedirs(odir)
    # write input arguments to file
    with open(os.path.join(odir,'input_params.csv'), 'w') as f:
        for x in args.__dict__.items():
            f.write("%s,%s\n"%(x[0],x[1]))
    # write images
    if model_recon=="monoexp":
        np.save(os.path.join(odir,'x_r'), x_r)
        np.save(os.path.join(odir,'gam_r'), gam_r)
    elif model_recon=="fixedcomp":
        np.save(os.path.join(odir,'x_r'), x_r)
    else:
        raise NotImplementedError
    print("Saved results and input parameters")


#-------------------------------------------------------------------------------------
# Initialize parameters for "raw" data
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
    fov = 220.
elif pathology=='CSF':
    casedir = f"CSF_H0{args.case}b1_birdcage"
    sdir1 = os.path.join(args.sdir, casedir, 'TE05','DeNoise')
    sdir2 = os.path.join(args.sdir, casedir, 'TE5', 'DeNoise')
    fpattern = '*.c?'
    ncoils  = 1
    data_n = 64
    te1 = 0.5
    delta_t = 4.5
    fov = 220. # TODO check if true

# find files
fnames1 = sorted(glob(os.path.join(sdir1, fpattern)))
fnames2 = sorted(glob(os.path.join(sdir2, fpattern)))
if (len(fnames1) != ncoils) or (len(fnames2) != ncoils):
    raise ValueError('The number of data files is not correct')
odir = os.path.join(sdir, casedir, 'results', f'betax_{beta_x:.1E}'+ (f'_betagam_{beta_gam:.1E}' if model_recon=='monoexp' else '')+
                                   (f'_{highres_mri_name}_nbow_{nnearest}' if not no_highres_prior else '')+
                                   (f'_t2bs_{t2bi_s:.1E}_t2bl_{t2bi_l:.1E}' if model_recon=='fixedcomp' else '') )

# high resolution proton MRI image
if not no_highres_prior:
    # already preprocessed .npy image (N4, registered)
    highresH_img_file = os.path.join(sdir, casedir, f"proton/{highres_mri_name}_n4_aligned_{n}.npy")
    # original nifti image without preprocessing
    highresH_img_file_nii = os.path.join(sdir, casedir, f"proton/{highres_mri_name}.nii")

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


#-------------------------------------------------------------------------------------
# Load and preprocess raw data
#-------------------------------------------------------------------------------------

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
sos_te1_for_sens = pynamr.sum_of_squares_reconstruction(data_pad[:,0]*filt, complex_format=True)

# compute sensitivities
for i in range(ncoils):
    coil_im = pynamr.simple_reconstruction(data_pad[i,0]*filt, complex_format=True)
    sens[i,...]   = coil_im / sos_te1_for_sens

# scale signal to be not too far away from 1
scale_factor = np.max(sos_te1_for_sens)
data = data / scale_factor
print("Loaded and preprocessed k-space data")
print("Computed spatial sensitivity maps")

# higher res proton image
if not no_highres_prior:
    if os.path.exists(highresH_img_file):
        # load the already preprocessed image
        highresH_img = np.load(highresH_img_file)
        print("Loaded already preprocessed higher res H prior image")
    else:
        print("Launched registration of higher res H prior image to simply reconstructed Na image")
        # load the original image
        highresH_img_nii     = nib.load(highresH_img_file_nii)
        highresH_img_nii     = nib.as_closest_canonical(highresH_img_nii)
        highresH_img         = highresH_img_nii.get_fdata()
        highresH_affine  = highresH_img_nii.affine
        highresH_voxsize = highresH_img_nii.header['pixdim'][1:4]
        highresH_origin  = highresH_affine[:-1,-1]

        # the images should be in the same orientation at least
        # empirical fix for nifti without proper orientation info
        if pathology=="CSF":
            highresH_img = np.flip(highresH_img, (0,1))

        # n4 correction
        highresH_img = pynamr.n4(highresH_img)

        # simple recon of Sodium TE1 image, interpolated to higher resolution
        sos_te1_for_reg = pynamr.sum_of_squares_reconstruction(data[:,0,...], complex_format=True)
        sos_te1_for_reg = zoom(sos_te1_for_reg, np.array(recon_shape)/ np.array(data_shape), order = 2, prefilter=False) 
        sos_te1_for_reg_voxsize = fov / np.array(recon_shape)
        sos_te1_for_reg_origin = np.full(3,-fov/2.)

        # register
        highresH_img = pynamr.register_highresH_to_lowresNa(highresH_img, sos_te1_for_reg, highresH_voxsize, sos_te1_for_reg_voxsize, highresH_origin, sos_te1_for_reg_origin)

        # save to npy file
        np.save(highresH_img_file, highresH_img)

        # check the registration
        vi_check_reg = pv.ThreeAxisViewer([highresH_img, sos_te1_for_reg, highresH_img],[None, None, sos_te1_for_reg], imshow_kwargs = {'cmap':plt.cm.Greys_r})
        print("Performed registration of higher res H prior image to simply reconstructed Na image and exited")

        # exit
        sys.exit()

#-------------------------------------------------------------------------------------
# "Standard/simple" recon
#-------------------------------------------------------------------------------------

# currently sum of squares of padded kspace data
std_te1 = pynamr.sum_of_squares_reconstruction(data_pad[:,0,...], complex_format=True)
std_te2 = pynamr.sum_of_squares_reconstruction(data_pad[:,1,...], complex_format=True)

# filter the SOS images
std_te1_filtered = gaussian_filter(std_te1, 1.)
std_te2_filtered = gaussian_filter(std_te2, 1.)

print("Computed simple reconstructions")

#-------------------------------------------------------------------------------------
# Check the preprocessed data before proceeding to recon and exit
#-------------------------------------------------------------------------------------

if check:
    vi_check_reg = pv.ThreeAxisViewer([highresH_img, std_te1, highresH_img],[None, None, std_te1], imshow_kwargs = {'cmap':plt.cm.Greys_r})
    if ncoils==8:
        sens_disp = np.abs(sens)
        vi_check_sens_1 = pv.ThreeAxisViewer([sens_disp[0], sens_disp[1], sens_disp[2], sens_disp[3]], imshow_kwargs = {'cmap':plt.cm.viridis})
        vi_check_sens_2 = pv.ThreeAxisViewer([sens_disp[4], sens_disp[5], sens_disp[6], sens_disp[7]], imshow_kwargs = {'cmap':plt.cm.viridis})
    print("Displayed preprocessed data for checking and exited")
    sys.exit()

#-------------------------------------------------------------------------------------
# Load and display already computed reconstruction results and exit
#-------------------------------------------------------------------------------------

if load_results:
    if model_recon == "monoexp":
        x_r = np.load(os.path.join(odir,'x_r.npy'))
        gam_r = np.load(os.path.join(odir,'gam_r.npy'))
        vi = display_results()
    elif model_recon == "fixedcomp":
        x_r = np.load(os.path.join(odir,'x_r.npy'))
        vi = display_results()
    print("Loaded and displayed previous reconstruction results and exited")
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
aimg = highresH_img.copy()
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
# run the recons
print("Started reconstruction")
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
    x_r = pynamr.complex_view_of_real_array(unknowns[pynamr.VarName.IMAGE].value * scale_factor)
    gam_r = unknowns[pynamr.VarName.GAMMA].value

    # save the results
    save_results()

    # show the results
    vi = display_results()

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
    x_r = pynamr.complex_view_of_real_array(unknowns[pynamr.VarName.PARAM].value * scale_factor)

    # save the results
    save_results()

    # show the results
    vi = display_results()

else:
    raise NotImplementedError



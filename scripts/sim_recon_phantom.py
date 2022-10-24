""" Phantom raw data simulation and reconstruction for dual echo TPI Sodium data

    Fixed compartmental T2* model:
    - knowns: short and long T2* components and their ratio for a biexponential compartment, long T2* component for a monoexponential compartment (fluid)
    - unknowns: "concentrations" for each compartment

    Monoexponential T2* model:
    - unknowns: monoexponential T2* map (Gamma = T2* decay between TE1 and TE2), "concentration"

    Implementation details: scipy.optimize.minimize requires flattened numpy arrays, has no cupy implementation, and complex arrays need to be converted to explicit real and imaginary dimensions


"""

import warnings

try:
    import cupy as cp
except ImportError:
    warnings.warn("cupy package not available", RuntimeWarning)
    

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import fmin_l_bfgs_b
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from copy import deepcopy
from argparse import ArgumentParser
import sys

import pynamr

parser = ArgumentParser(description = '3D phantom Na MRI dual echo simulation and reconstruction')
parser.add_argument('--niter',  default = 20, type = int, help='number of optimization iterations')
parser.add_argument('--n_outer', default = 10, type = int, help='number of outer optimization iterations if multistep')
parser.add_argument('--beta_x', default = 1e-2, type = float, help='penalty strength for image/param')
parser.add_argument('--beta_gam', default = 1e-2, type = float, help='penalty strength for T2* decay')
parser.add_argument('--n', default = 128, type = int, choices = [128,256], help='image size')
parser.add_argument('--n_readout_bins', default = 16, type = int, help='TPI readout bins')
parser.add_argument('--noise_level', default = 5.,  type = float, help='Gaussian noise level')
parser.add_argument('--nnearest', default = 13,  type = int, help='Bowsher number of most similar voxels')
parser.add_argument('--phantom',  default = 'rod', choices = ['rod'], help='phantom type')
parser.add_argument('--seed',     default = 1, type = int, help='seed for random generator')
parser.add_argument('--delta_t', default = 4.7, type = float, help='Time between TE1 and TE2 acquisition')
parser.add_argument('--te1', default = 0.3, type = float, help='TE1, start of the first acquisition')
parser.add_argument('--ncoils',   default = 1, type = int, help='number of coils')
parser.add_argument('--data_n',   default = 64, type = int, help='data size')
parser.add_argument('--instant_tpi_recon',   default = False, action='store_true', help='TPI readout instantaneous for reconstruction')
parser.add_argument('--instant_tpi_sim',   default = False, action='store_true', help='TPI readout instantaneous for simulating raw data')
parser.add_argument('--model_sim',   default = 'monoexp', type = str, choices = ['monoexp','fixedcomp','custom'],
                       help='forward model for simulating raw data')
parser.add_argument('--model_recon',   default = 'monoexp', type = str, choices = ['monoexp','fixedcomp','custom'],
                       help='forward model for reconstructing the raw data')
parser.add_argument('--model_im',   default = 'monoexp', type = str, choices = ['monoexp','fixedcomp','custom'],
                       help='model for building the true image')
parser.add_argument('--t2bi_s', default = 3., type = float, help='fixed T2* biexponential short component')
parser.add_argument('--t2bi_l', default = 20., type = float, help='fixed T2* biexponential long component')
parser.add_argument('--t2mono_l', default = 25., type = float, help='fixed T2* monoexponential component (i.e. fluid)')
parser.add_argument('--t2bi_frac_l', default = 0.4, type = float, help='fixed T2* biexponential fraction of long component')
parser.add_argument('--t2bi_smap_filename', default = None, type = str, help='T2* biexponential short component spatial map')
parser.add_argument('--t2bi_lmap_filename', default = None, type = str, help='T2* biexponential long component spatial map')
parser.add_argument('--t2mono_map_filename', default = None, type = str, help='T2* monoexponential (long) component spatial map (i.e. fluid)')
parser.add_argument('--only_sim', default = False, action='store_true', help='only simulate raw data')
parser.add_argument('--only_sim_simplerecon', default = False, action='store_true', help='only simulate raw data and std recon')
parser.add_argument('--save_results', default = False, action='store_true', help='save results to binary files .img')
parser.add_argument('--results_dir', default = 'results/', help='folder for saving results')
parser.add_argument('--hann_simplerecon', default = False, action='store_true', help='apply hanning filter for standard recon')



args = parser.parse_args()

niter       = args.niter
n_outer     = args.n_outer
beta_x      = args.beta_x
beta_gam    = args.beta_gam
n           = args.n
n_readout_bins     = args.n_readout_bins
noise_level = args.noise_level
nnearest    = args.nnearest
phantom     = args.phantom
seed        = args.seed
ncoils      = args.ncoils
delta_t     = args.delta_t
te1         = args.te1
data_n      = args.data_n
instant_tpi_recon = args.instant_tpi_recon
instant_tpi_sim = args.instant_tpi_sim
model_im    = args.model_im
model_sim   = args.model_sim
model_recon = args.model_recon
t2bi_s      = args.t2bi_s
t2bi_l      = args.t2bi_l
t2bi_frac_l = args.t2bi_frac_l
t2mono_l    = args.t2mono_l
t2bi_smap_filename = args.t2bi_smap_filename
t2bi_lmap_filename = args.t2bi_lmap_filename
t2mono_map_filename = args.t2mono_map_filename
only_sim    = args.only_sim
only_sim_simplerecon = args.only_sim_simplerecon
save_results = args.save_results
results_dir = args.results_dir
hann_simplerecon   = args.hann_simplerecon

#-------------------------------------------------------------------------------------
# initialize some parameters
noiseless = (not (noise_level>0.))
# 2 TE values currently
nechos = 2
# shape of the kspace data of a single coil
data_shape = (data_n, data_n, data_n)
# down sample factor (recon cube size / data cube size), must be integer
ds = round(n/data_n)
# create complex sensitivity images - TO BE IMPROVED
sens = np.ones((ncoils, n, n, n)) + 0j * np.zeros(
    (ncoils, n, n, n))
# seed the random generator
np.random.seed(seed)

#-------------------------------------------------------------------------------------
# setup the base phantom
if phantom=='rod':
    # oversampling factor used to generate the phantom
    osf = 6
    # generate oversampled phantom
    x_ph, gam_ph = pynamr.rod_phantom(n=osf * n)
    # downsample phantom (along each of the 3 axis)
    x_ph = pynamr.downsample(pynamr.downsample(pynamr.downsample(x_ph, osf, axis=0),
                                                osf,
                                                axis=1),
                              osf,
                              axis=2)
    gam_ph = pynamr.downsample(pynamr.downsample(pynamr.downsample(gam_ph, osf, axis=0),
                                                  osf,
                                                  axis=1),
                                osf,
                                axis=2)
    gam_ph /= gam_ph.max()

    if model_im=="fixedcomp":
        # biexpo and monoexpo "concentrations" for the fixed compartmental T2* model
        x1 = 0.5*x_ph
        x2 = 0.5*np.swapaxes(x_ph,0,1)

        # corresponding "true" TE1, TE2 and Gamma images 
        true_conc = x1 + x2
        true_te2 = x1 * ( (1-t2bi_frac_l) * np.exp(-(delta_t+te1)/t2bi_s) + t2bi_frac_l * np.exp(-(delta_t+te1)/t2bi_l) ) + x2 * np.exp(-(delta_t+te1)/t2mono_l)
        true_te1 = x1 * ( (1-t2bi_frac_l) * np.exp(-te1/t2bi_s) + t2bi_frac_l * np.exp(-te1/t2bi_l) ) + x2 * np.exp(-te1/t2mono_l)

        # true_te1 * true_gam = true_te2, true_te1<=0.05 is the 0 background
        true_gam = np.divide(true_te2,true_te1, where=(true_te1>1e-05))
        true_gam[true_te1<=1e-05]=1.

else:
    raise NotImplementedError


#-------------------------------------------------------------------------------------
# Simulation of raw data
#-------------------------------------------------------------------------------------

# readout config
if instant_tpi_sim:
    readout_time_sim = pynamr.TPIInstantaneousReadOutTime()
else:
    readout_time_sim = pynamr.TPIReadOutTime()

# k-space config
kspace_part = pynamr.RadialKSpacePartitioner(data_shape, n_readout_bins)

#-------------------------------------------------------------------------------------
# forward model for simulating raw data
if model_sim == "monoexp":
    # construct true images
    if model_im == "fixedcomp":
        x = true_conc
        # add imaginary dimension
        x = np.stack([x, 0 * x], axis=-1)
        gam = true_gam
        aimg = true_conc
    elif model_im == "monoexp":
        x = x_ph
        # add imaginary dimension
        x = np.stack([x, 0 * x], axis=-1)
        gam = gam_ph
        true_gam = gam_ph
        true_conc = x_ph
        true_te1 = true_conc * gam**(te1/delta_t)
        true_te2 = true_te1 * gam
        aimg = x_ph
    else:
        raise NotImplementedError

    # forward model and unknown variables for simulating raw data
    fwd_model_sim = pynamr.MonoExpDualTESodiumAcqModel(ds, sens, delta_t, te1, readout_time_sim, kspace_part)
    unknowns_sim = {pynamr.VarName.IMAGE: pynamr.Var(shape=tuple([ds * x for x in data_shape]) + (2,)),
                    pynamr.VarName.GAMMA: pynamr.Var(shape=tuple([ds * x for x in data_shape]), nb_comp=1, complex_var=False)}
    unknowns_sim[pynamr.VarName.IMAGE].value = x
    unknowns_sim[pynamr.VarName.GAMMA].value = gam

elif model_sim == "fixedcomp":
    if model_im == "fixedcomp":
        x = np.stack([x1, x2], axis=0)
        # add imaginary dimension
        x = np.stack([x, 0 * x], axis=-1)
        aimg = x1

        if save_results:
            x1.tofile(results_dir+'true_comp0.img')
            x2.tofile(results_dir+'true_comp1.img')
    else:
        raise NotImplementedError

    # forward model and unknown variables for simulating raw data
    fwd_model_sim = pynamr.TwoCompartmentBiExpDualTESodiumAcqModel(ds, sens, delta_t, te1, readout_time_sim, kspace_part,  0, t2mono_l, t2bi_s, t2bi_l, 1, t2bi_frac_l)
    unknowns_sim = {pynamr.VarName.PARAM: pynamr.Var(shape=tuple([2,] + [ds * x for x in data_shape] + [2,]), nb_comp=2)}
    unknowns_sim[pynamr.VarName.PARAM].value = x

# save true images if required
if save_results:
    true_conc.tofile(results_dir+'true_conc.img')
    true_te1.tofile(results_dir+'true_te1.img')
    true_te2.tofile(results_dir+'true_te2.img')
    true_gam.tofile(results_dir+'true_gam.img')


#-------------------------------------------------------------------------------------
# generate data
y = fwd_model_sim.forward(unknowns_sim)

# add noise
if noiseless:
    data = y
else:
    #data = y + noise_level * np.abs(y).mean() * np.random.randn(*y.shape).astype(np.float64)
    # temporary, to ensure we have the same noise level computation for different options
    data = y + noise_level * 0.014 * np.random.randn(*y.shape).astype(np.float64)

#-------------------------------------------------------------------------------------
# only simulate raw data end exit
if only_sim:
    sys.exit()


#-------------------------------------------------------------------------------------
# "Standard/simple" recon
#-------------------------------------------------------------------------------------

# currently sum of squares, TODO implement a better std recon
std_te1 = pynamr.sum_of_squares_reconstruction(data[:,0,...])
std_te2 = pynamr.sum_of_squares_reconstruction(data[:,1,...])

# upsample to recon size
std_te1 = pynamr.upsample_nearest(pynamr.upsample_nearest(pynamr.upsample_nearest(std_te1, ds, axis=0), ds, axis=1), ds, axis=2)
std_te2 = pynamr.upsample_nearest(pynamr.upsample_nearest(pynamr.upsample_nearest(std_te2, ds, axis=0), ds, axis=1), ds, axis=2)

# produce filtered simple recon
if hann_simplerecon:
    # hann filter and zero padding
    # half data size
    data_n_half = data_n//2

    # multiplicative filter in fourier space
    h_win = interp1d(np.arange(data_n_half), np.hanning(data_n)[data_n_half:], fill_value = 0, bounds_error = False)
    k0,k1,k2 = np.meshgrid(np.arange(-data_n_half,data_n_half),np.arange(-data_n_half,data_n_half),np.arange(-data_n_half,data_n_half))
    abs_k = np.sqrt(k0**2 + k1**2 + k2**2)
    hmask = h_win(abs_k.flatten()).reshape(data_n,data_n,data_n)
    hmask = np.fft.fftshift(hmask)

    # zero padded and filtered complex data
    data_filt = np.zeros((ncoils,nechos,n,n,n), np.complex128)
    # padding size
    dif = (n-data_n)//2
    # normalization factor between low and high res fft
    factor_fft_res_change = 2*np.sqrt([2])
    for c in range(ncoils):
        for e in range(nechos):
            data_filt[c,e,:,:,:] = np.fft.fftshift(np.pad(np.fft.fftshift(hmask*pynamr.complex_view_of_real_array(data[c,e])), dif)*factor_fft_res_change)

    # simple recon of zero padded and filtered data
    std_te1_filtered = pynamr.sum_of_squares_reconstruction(data_filt[:,0], complex_format=True)
    std_te2_filtered = pynamr.sum_of_squares_reconstruction(data_filt[:,1], complex_format=True)
else:
    # gaussian image filter
    std_te1_filtered = gaussian_filter(std_te1, 1.)
    std_te2_filtered = gaussian_filter(std_te2, 1.)

# save images if required
if save_results:
    std_te1.tofile(results_dir+'std_te1_im'+model_im+'_sim'+model_sim+('_inst' if instant_tpi_sim else '')+('_noiseless' if noiseless else '')+'.img')
    std_te2.tofile(results_dir+'std_te2_im'+model_im+'_sim'+model_sim+('_inst' if instant_tpi_sim else '')+('_noiseless' if noiseless else '')+'.img')
    std_te1_filtered.tofile(results_dir+'std_te1_filt_im'+model_im+'_sim'+model_sim+('_inst' if instant_tpi_sim else '')+('_noiseless' if noiseless else '')+('_hann' if hann_simplerecon else '')+'.img')
    std_te2_filtered.tofile(results_dir+'std_te2_filt_im'+model_im+'_sim'+model_sim+('_inst' if instant_tpi_sim else '')+('_noiseless' if noiseless else '')+('_hann' if hann_simplerecon else '')+'.img')

# only simulate raw data and simple recon
if only_sim_simplerecon:
    sys.exit()

#-------------------------------------------------------------------------------------
# Reconstruction
#-------------------------------------------------------------------------------------

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
aimg = (aimg.max() - aimg)**0.5
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

    x_r = unknowns[pynamr.VarName.IMAGE].value
    gam_r = unknowns[pynamr.VarName.GAMMA].value

    # show the results
    ims_1 = dict(cmap=plt.cm.viridis, vmin = 0, vmax = true_conc.max())
    ims_2 = dict(cmap=plt.cm.viridis, vmin = 0, vmax = true_gam.max())

    import pymirc.viewer as pv
    vi = pv.ThreeAxisViewer([true_conc,
                         np.linalg.norm(x_r, axis=-1),
                         std_te1_filtered,
                         true_gam,
                         gam_r],
                         imshow_kwargs=[ims_1,ims_1,ims_1,ims_2,ims_2])
    if save_results:
        x_r.tofile(results_dir+'x_r_im'+model_im+'_sim'+model_sim+('_inst' if instant_tpi_sim else '')+'_rec'+model_recon+('_inst' if instant_tpi_recon else '')+('_noiseless' if noiseless else '')+'.img')
        gam_r.tofile(results_dir+'gam_r_im'+model_im+'_sim'+model_sim+('_inst' if instant_tpi_sim else '')+'_rec'+model_recon+('_inst' if instant_tpi_recon else '')+('_noiseless' if noiseless else '')+'.img')



elif model_recon=="fixedcomp":
    # allocate arrays for recons and copy over initial values
    unknowns[pynamr.VarName.PARAM].value = np.ones(unknowns[pynamr.VarName.PARAM].shape, np.float64)

    # update complex sodium image
    res_1 = fmin_l_bfgs_b(loss,
                          unknowns[pynamr.VarName.PARAM].value.copy().ravel(),
                          fprime=loss.gradient,
                          args=(unknowns, pynamr.VarName.PARAM),
                          maxiter=niter,
                          disp=1)

    unknowns[pynamr.VarName.PARAM].value = res_1[0].copy().reshape(unknowns[pynamr.VarName.PARAM].shape)
    x_r = unknowns[pynamr.VarName.PARAM].value

    # show the results
    ims_1 = dict(cmap=plt.cm.viridis, vmin = 0, vmax = x.max())
    import pymirc.viewer as pv
    vi = pv.ThreeAxisViewer([x1,
                         np.linalg.norm(x_r[0], axis=-1),
                         x2,
                         np.linalg.norm(x_r[1], axis=-1)],
                         imshow_kwargs=[ims_1, ims_1, ims_1, ims_1])

    if save_results:
        x_r.tofile(results_dir+'x_r_im'+model_im+'_sim'+model_sim+('_inst' if instant_tpi_sim else '')+'_rec'+model_recon+('_inst' if instant_tpi_recon else '')+('_noiseless' if noiseless else '')+'.img')

else:
    raise NotImplementedError




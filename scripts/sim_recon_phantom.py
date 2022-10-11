""" Phantom raw data simulation and reconstruction for dual echo TPI Sodium data

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
from scipy.ndimage import gaussian_filter
from scipy.optimize import fmin_l_bfgs_b
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
parser.add_argument('--delta_t', default = 4.5, type = float, help='Time between TE1 and TE2 acquisition')
parser.add_argument('--te1', default = 0.5, type = float, help='TE1, start of the first acquisition')
parser.add_argument('--ncoils',   default = 1, type = int, help='number of coils')
parser.add_argument('--data_n',   default = 64, type = int, help='data size')
parser.add_argument('--instant_tpi',   default = False, action='store_true', help='TPI readout instantaneous')
parser.add_argument('--model_sim',   default = 'monoexp', type = str, choices = ['monoexp','fixed_comp','custom'],
                       help='forward model for simulating raw data')
parser.add_argument('--model_recon',   default = 'monoexp', type = str, choices = ['monoexp','fixed_comp','custom'],
                       help='forward model for reconstructing the raw data')
parser.add_argument('--model_im',   default = 'monoexp', type = str, choices = ['monoexp','fixed_comp','custom'],
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
instant_tpi = args.instant_tpi
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

#-------------------------------------------------------------------------------------
# initialize some parameters

# 2 TE values currently
nechos = 2
# shape of the kspace data of a single coil
data_shape = (data_n, data_n, data_n)
# down sample factor (recon cube size / data cube size), must be integer
ds = round(n/data_n)
# create sensitivity images - TO BE IMPROVED
sens = np.ones((ncoils, data_n, data_n, data_n)) + 0j * np.zeros(
    (ncoils, data_n, data_n, data_n))
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

    if model_im=="fixed_comp":
        # biexpo and monoexpo "concentrations" for the fixed compartmental T2* model
        x1 = 0.5*x_ph
        x2 = 0.5*np.swapaxes(x_ph,0,1)

        # corresponding "true" TE1, TE2 and Gamma images 
        true_te1 = x1 + x2
        true_te2 = x1 * ( (1-t2bi_frac_l) * np.exp(-delta_t/t2bi_s) + t2bi_frac_l * np.exp(-delta_t/t2bi_l) ) + x2 * np.exp(-delta_t/t2mono_l)

        # true_te1 * true_gam = true_te2, true_te1<=0.05 is the 0 background
        true_gam = np.divide(true_te2,true_te1, where=(true_te1>1e-05))
        true_gam[true_te1<=1e-05]=1.

else:
    raise NotImplementedError

#-------------------------------------------------------------------------------------
# construct the forward model for simulating the raw data and generate data 

# readout config
if instant_tpi:
    readout_time = pynamr.TPIInstantaneousReadOutTime()
else:
    readout_time = pynamr.TPIReadOutTime()

# k-space config
kspace_part = pynamr.RadialKSpacePartitioner(data_shape, n_readout_bins)

# forward model for simulating raw data
if model_sim == "monoexp":
    # construct true images
    if model_im == "fixed_comp":
        x = true_te1
        # add imaginary dimension
        x = np.stack([x, 0 * x], axis=-1)
        gam = true_gam
    elif model_im == "monoexp":
        x = x_ph
        # add imaginary dimension
        x = np.stack([x, 0 * x], axis=-1)
        gam = gam_ph
    else:
        raise NotImplementedError

    # forward model and unknown variables for simulating raw data
    fwd_model_sim = pynamr.MonoExpDualTESodiumAcqModel(ds, sens, delta_t, te1, readout_time, kspace_part)
    unknowns_sim = {pynamr.VarName.IMAGE: pynamr.Var(shape=tuple([ds * x for x in data_shape]) + (2,)),
                    pynamr.VarName.GAMMA: pynamr.Var(shape=tuple([ds * x for x in data_shape]), nb_comp=1, complex_var=False)}
    unknowns_sim[pynamr.VarName.IMAGE].value = x
    unknowns_sim[pynamr.VarName.GAMMA].value = gam

elif model_sim == "fixed_comp":
    if model_im == "fixed_comp":
        x = np.stack([x1, x2], axis=0)
        # add imaginary dimension
        x = np.stack([x, 0 * x], axis=-1)
    else:
        raise NotImplementedError

    # forward model and unknown variables for simulating raw data
    fwd_model_sim = pynamr.TwoCompartmentBiExpDualTESodiumAcqModel(ds, sens, delta_t, te1, readout_time, kspace_part,  0, t2mono_l, t2bi_s, t2bi_l, 1, t2bi_frac_l)
    unknowns_sim = {pynamr.VarName.PARAM: pynamr.Var(shape=tuple([2,] + [ds * x for x in data_shape] + [2,]), nb_comp=2)}
    unknowns_sim[pynamr.VarName.PARAM].value = x


# generate data
y = fwd_model_sim.forward(unknowns_sim)
# add noise
if noise_level>0.:
    data = y + noise_level * np.abs(y).mean() * np.random.randn(*y.shape).astype(np.float64)
else:
    data = y

# only simulate raw data
if only_sim:
    sys.exit()

#-------------------------------------------------------------------------------------
# perform "standard" reconstructions, sum of squares currently,
# TODO implement a better std recon
std_te1 = pynamr.sum_of_squares_reconstruction(data[:,0,...])
std_te2 = pynamr.sum_of_squares_reconstruction(data[:,1,...])

# filter the SOS images and upsample to recon grid
std_te1_filtered = pynamr.upsample(pynamr.upsample(pynamr.upsample(gaussian_filter(std_te1, 1.), ds, 0), ds, 1), ds, 2)
std_te2_filtered = pynamr.upsample(pynamr.upsample(pynamr.upsample(gaussian_filter(std_te2, 1.), ds, 0), ds, 1), ds, 2)

# add imaginary dimension for consistency
std_te1 = np.stack([std_te1, 0 * std_te1], axis=-1)
std_te2 = np.stack([std_te2, 0 * std_te2], axis=-1)
std_te1_filtered = np.stack([std_te1_filtered, 0 * std_te1_filtered], axis=-1)
std_te2_filtered = np.stack([std_te2_filtered, 0 * std_te2_filtered], axis=-1)

# only simulate raw data and simple recon
if only_sim_simplerecon:
    sys.exit()

#-------------------------------------------------------------------------------------
# forward model and unknowns for reconstruction
if model_recon == "monoexp":
    fwd_model = pynamr.MonoExpDualTESodiumAcqModel(ds, sens, delta_t, te1, readout_time, kspace_part)
    unknowns = {pynamr.VarName.IMAGE: pynamr.Var(shape=tuple([ds * x for x in data_shape]) + (2,)),
                    pynamr.VarName.GAMMA: pynamr.Var(shape=tuple([ds * x for x in data_shape]), nb_comp=1, complex_var=False)}
elif model_recon == "fixed_comp":
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
aimg = x[...,0]
aimg = (aimg.max() - aimg)**0.5
bowsher_loss = pynamr.generate_bowsher_loss(aimg, nnearest)

#-------------------------------------------------------------------------------------
# setup the total loss function consiting of data fidelity loss and the priors
if model_recon == "monoexp":
    penalty_info = {pynamr.VarName.IMAGE: bowsher_loss, pynamr.VarName.GAMMA: bowsher_loss}
    beta_info = {pynamr.VarName.IMAGE: beta_x, pynamr.VarName.GAMMA: beta_gam}
elif model_recon == "fixed_comp":
    penalty_info = {pynamr.VarName.PARAM: bowsher_loss}
    beta_info = {pynamr.VarName.PARAM: beta_x}
else:
    raise NotImplementedError

loss = pynamr.TotalLoss(data_fidelity_loss, penalty_info, beta_info)

#-------------------------------------------------------------------------------------
# run the recons
if model_recon=="monoexp":
    # allocate initial values of unknown variables
    unknowns[pynamr.VarName.IMAGE].value = std_te1_filtered
    unknowns[pynamr.VarName.GAMMA].value = np.clip(std_te2_filtered[...,0] / (std_te1_filtered[...,0] + 1e-7), 0, 1)

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
    ims_1 = dict(cmap=plt.cm.viridis, vmin = 0, vmax = 1.2)
    ims_2 = dict(cmap=plt.cm.viridis, vmin = 0, vmax = 1.1*np.percentile(np.linalg.norm(std_te1_filtered, axis=-1),95))
    ims_3 = dict(cmap=plt.cm.viridis, vmin = 0, vmax = 1.)

    import pymirc.viewer as pv
    vi = pv.ThreeAxisViewer([np.linalg.norm(x, axis=-1),
                         np.linalg.norm(std_te1_filtered, axis=-1),
                         np.linalg.norm(x_r, axis=-1),
                         gam,
                         gam_r],
                         imshow_kwargs=[ims_1,ims_2,ims_1,ims_3,ims_3])

elif model_recon=="fixed_comp":
    # allocate arrays for recons and copy over initial values
    unknowns[pynamr.VarName.PARAM].value = np.ones(unknowns[pynamr.VarName.PARAM].shape, np.float64)

    # alternating LBFGS steps
    for i_it in range(niter):

        # update complex sodium image
        res_1 = fmin_l_bfgs_b(loss,
                          unknowns[pynamr.VarName.PARAM].value.copy().ravel(),
                          fprime=loss.gradient,
                          args=(unknowns, pynamr.VarName.PARAM),
                          disp=1)

        unknowns[pynamr.VarName.PARAM].value = res_1[0].copy().reshape(unknowns[pynamr.VarName.PARAM].shape)
        x_r = unknowns[pynamr.VarName.PARAM]


    # show the results
    ims_1 = dict(cmap=plt.cm.viridis, vmin = 0, vmax = 1.1*x.max())
    import pymirc.viewer as pv
    vi = pv.ThreeAxisViewer([x1,
                         np.linalg.norm(x_r[0], axis=-1),
                         x2,
                         np.linalg.norm(x_r[1], axis=-1)],
                         imshow_kwargs=[ims_1, ims_1, ims_1, ims_1])
else:
    raise NotImplementedError


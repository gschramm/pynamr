""" simple script to show how to jointly reconstruct simulated "inverse crime" dual echo sodium data
"""

import warnings

try:
    import cupy as cp
except ImportError:
    warnings.warn("cupy package not available", RuntimeWarning)
    

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import fmin_l_bfgs_b
from copy import deepcopy

import pynamr

#-------------------------------------------------------------------------------------
# input parameters

# seed for random generator
np.random.seed(1)
# shape of the kspace data of a single coil
data_shape = (64, 64, 64)
# down sample factor (recon cube size / data cube size), must be integer
ds = 2
# number of coils to simulate
ncoils = 4
# time (ms) between first and second echo, first echo is assumed to be a t=0
dt = 5.
# noise level
# realistic noise level to get SNR ca 5 in cylinder phantom with Na conc. 1
noise_level = 5.
# numpy/cupy module to use to calculate all FFTs (use cupy if on a GPU)
xp = np
# number of nearest neighbors for the Bowsher prior
nnearest = 13
# prior weight applied to real and imag part of complex sodium image
beta_x = 1e-2
# prior weight applied to the gamma (decay) image
#beta_gam = 1e-1
# number of outer LBFGS iterations
n_outer = 5 
# number of inner LBFGS iterations
n_inner = 10
# number of readout bins
n_readout_bins = 16

#-------------------------------------------------------------------------------------

n_ds = data_shape[0]
n = ds * n_ds

#-------------------------------------------------------------------------------------
# setup the phantoms

# oversampling factor used to generate the phantom
osf = 6

# generate oversampled phantom
x, gam = pynamr.rod_phantom(n=osf * n)

# downsample phantom (along each of the 3 axis)
x = pynamr.downsample(pynamr.downsample(pynamr.downsample(x, osf, axis=0),
                                        osf,
                                        axis=1),
                      osf,
                      axis=2)

#gam = pynamr.downsample(pynamr.downsample(pynamr.downsample(gam,
#                                                            osf,
#                                                            axis=0),
#                                          osf,
#                                          axis=1),
#                        osf,
#                        axis=2)
#gam /= gam.max()

# create pseudo-complex data and add dimension for "compartments"
x = np.stack([x, 0 * x], axis=-1)
x = np.stack([x, 0.5*x], axis=0)

# create sensitivity images - TO BE IMPROVED
sens = np.ones((ncoils, n_ds, n_ds, n_ds)) + 0j * np.zeros(
    (ncoils, n_ds, n_ds, n_ds))

#-------------------------------------------------------------------------------------
# generate a mono exp. acquisition model for 2 echoes
readout_time = pynamr.TPIReadOutTime()
kspace_part = pynamr.RadialKSpacePartitioner(data_shape, n_readout_bins)

# unknowns
unknowns = {pynamr.VarName.PARAM: pynamr.Var( shape=tuple([2,] + [ds * x for x in data_shape] + [2,]), nb_comp=2)}
unknowns[pynamr.VarName.PARAM].value = x

fwd_model = pynamr.TwoCompartmentBiExpDualTESodiumAcqModel(ds, sens, dt, readout_time, kspace_part, 2, 20, 4, 16, 0.4, 0.2)

# generate data
y = fwd_model.forward(unknowns)
data = y + noise_level * np.abs(y).mean() * np.random.randn(*y.shape).astype(np.float64)

#-------------------------------------------------------------------------------------
# setup the data fidelity loss function
data_fidelity_loss = pynamr.DataFidelityLoss(fwd_model, data)

#-------------------------------------------------------------------------------------
# setup the priors

# simulate a perfect anatomical prior image (with changed contrast but matching edges)
aimg = x[0,..., 0]
aimg = (aimg.max() - aimg)**0.5

# define neighborhood where to look for neasrest Bowsher neighbors
s = np.array([[[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]],
              [[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]],
              [[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1],
               [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]],
              [[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]],
              [[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]])

# generate the neasrest "Bowsher" neighbors
nn_inds = np.zeros((np.prod(aimg.shape), nnearest), dtype=np.uint32)

# setup anatomical or non-anatomical prior
# nearest neighbors anatomical Bowsher prior
# next neighbors is non-anatomical priors

# anatomical prior
pynamr.nearest_neighbors(aimg, s, nnearest, nn_inds)

# non-anatomical prior
#pynamr.next_neighbors(aimg.shape, nn_inds)

# "adjoint" list of nearest/next neighbors
nn_inds_adj = pynamr.is_nearest_neighbor_of(nn_inds)

bowsher_loss = pynamr.BowsherLoss(nn_inds, nn_inds_adj)

#-------------------------------------------------------------------------------------
# setup the total loss function consisting of data fidelity loss and the priors on the
# sodium images
penalty_info = {pynamr.VarName.PARAM: bowsher_loss}
beta_info = {pynamr.VarName.PARAM: beta_x}
loss = pynamr.TotalLoss(data_fidelity_loss, penalty_info, beta_info)

#-------------------------------------------------------------------------------------
# run the recons

# initialize recons
x_0 = 0 * x + 1

# allocate arrays for recons and copy over initial values
unknowns[pynamr.VarName.PARAM].value = x_0.copy()

#------------------
# alternating LBFGS steps
for i_out in range(n_outer):

    # update complex sodium image
    res_1 = fmin_l_bfgs_b(loss,
                          unknowns[pynamr.VarName.PARAM].value.copy().ravel(),
                          fprime=loss.gradient,
                          args=(unknowns, pynamr.VarName.PARAM),
                          maxiter=n_inner,
                          disp=1)

    unknowns[pynamr.VarName.PARAM].value = res_1[0].copy().reshape(unknowns[pynamr.VarName.PARAM].shape)

#------------------

x_r = unknowns[pynamr.VarName.PARAM].value

""" simple script to show how to jointly reconstruct simulated "inverse crime" dual echo sodium data
"""

# comment if cupy/GPU is not available + use xp = np below
import cupy as cp
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import fmin_l_bfgs_b

import pynamr

#-------------------------------------------------------------------------------------
# input parameters

# seed for random generator
np.random.seed(1)
# shape of the kspace data of a single coil
data_shape  = (64,64,64)
# down sample factor (recon cube size / data cube size), must be integer
ds          = 2
# number of coils to simulate
ncoils      = 4
# time (ms) between first and second echo, first echo is assumed to be a t=0
dt          = 5.
# noise level
# realistic noise level to get SNR ca 5 in cylinder phantom with Na conc. 1
noise_level = 5. 
# numpy/cupy module to use to caluclate all FFTs (use cupy if on a GPU)
xp          = cp
# number of neasrest neighbors for the Bowhser prior
nnearest    = 13
# prior weight applied to real and imag part of complex sodium image
beta_x      = 1e-2
# prior weight applied to the gamma (decay) image
beta_gam    = 1e-2
# number of outer LBFGS iterations
n_outer     = 20
# number of inner LBFGS iterations
n_inner     = 50

#-------------------------------------------------------------------------------------

n_ds = data_shape[0] 
n    = ds*n_ds

#-------------------------------------------------------------------------------------
# setup the phantoms

# oversampling factor used to generate the phantom
osf = 6

# generate oversampled phantom
x, gam = pynamr.rod_phantom(n = osf*n)

# downsample phantom
x = pynamr.downsample(pynamr.downsample(pynamr.downsample(x, osf, np, axis = 0), osf, np, axis = 1), osf, np, axis = 2)
gam = pynamr.downsample(pynamr.downsample(pynamr.downsample(gam, osf, np, axis = 0), osf, np, axis = 1), osf, np, axis = 2)
gam /= gam.max()

# create pseudo-complex data
x = np.stack([x, 0*x], axis = -1)

# create sensitivity images - TO BE IMPROVED
sens = np.ones((ncoils,n_ds,n_ds,n_ds)) + 0j*np.zeros((ncoils,n_ds,n_ds,n_ds))


#-------------------------------------------------------------------------------------
# generate a mono exp. acquisition model for 2 echoes
fwd_model = pynamr.MonoExpDualTESodiumAcqModel(data_shape, ds, ncoils, sens, dt, xp)

# generate data
y    = fwd_model.forward(x, gam)
data = y + noise_level*np.abs(y).mean()*np.random.randn(*y.shape).astype(np.float64)


#d = np.squeeze(data.astype(np.float64).view(dtype = np.complex128), axis = -1)
#
#a = np.fft.ifftn(d[0,0,...], norm = 'ortho')
#b = np.fft.ifftn(d[0,1,...], norm = 'ortho')
#
#q  = np.abs(a)
#qs = qs = gaussian_filter(q,3)
#mask = (qs > 0.9)

#-------------------------------------------------------------------------------------
# setup the data fidelity loss fucntion
data_fidelity_loss = pynamr.DataFidelityLoss(fwd_model, data)

#-------------------------------------------------------------------------------------
# setup the priors

# simulate a perfect anatomical prior image (with changed contrast but matching edges)
aimg     = x[...,0]
aimg     = (aimg.max() - aimg)**0.5

# define neighborhood where to look for neasrest Bowsher neighbors
s = np.array([[[0, 0, 0, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0]],
              [[0, 1, 1, 1, 0],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [0, 1, 1, 1, 0]],
              [[0, 1, 1, 1, 0],
               [1, 1, 1, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 1, 1, 1, 1],
               [0, 1, 1, 1, 0]],
              [[0, 1, 1, 1, 0],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [0, 1, 1, 1, 0]],
              [[0, 0, 0, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0]]])


# generate the neasrest "Bowsher" neighbors
nn_inds  = np.zeros((np.prod(aimg.shape), nnearest), dtype = np.uint32)

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
# setup the total loss function consiting of data fidelity loss and the priors on the
# sodium and gamma images

loss = pynamr.TotalLoss(data_fidelity_loss, bowsher_loss, bowsher_loss, beta_x, beta_gam)

#-------------------------------------------------------------------------------------
# run the recons

# initialize recons
x_0   = 0*x + 1
gam_0 = 0*gam + 1

# allocate arrays for recons and copy over initial values
x_r   = x_0.copy()
gam_r = gam_0.copy()

#------------------
# alternating LBFGS steps
for i_out in range(n_outer):

   # update complex sodium image
  res_1 = fmin_l_bfgs_b(loss.eval_x_first, x_r, fprime = loss.grad_x, 
                            args = (gam_r,), maxiter = n_inner, disp = 1)
  x_r = res_1[0].copy()

  # update real gamma (decay) image
  res_2 = fmin_l_bfgs_b(loss.eval_gam_first, gam_r, fprime = loss.grad_gam, 
                            args = (x_r,), maxiter = n_inner, disp = 1,
                            bounds = (gam_0.size)*[(0.001,1)])

  gam_r = res_2[0].copy()

#------------------

x_r = x_r.reshape(x.shape)
gam_r = gam_r.reshape(gam.shape)

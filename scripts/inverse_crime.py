import cupy as cp
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import fmin_l_bfgs_b

import pynamr

# input parameters
np.random.seed(1)
data_shape  = (64,64,64)
ds          = 2
ncoils      = 4
dt          = 5.
noise_level = 0
xp          = cp

nnearest    = 13
beta_x      = 1e-2
beta_gam    = 1e-2

n_outer     = 10
n_inner     = 50

n_ds = data_shape[0] 
n    = ds*n_ds

#-------------------------------------------------------------------------------------
# setup the phantoms
osf = 6

# generate oversampled phantom
x, gam = pynamr.rod_phantom(n = osf*n)

# downsample phantom
x = pynamr.downsample(pynamr.downsample(pynamr.downsample(x, osf, np, axis = 0), osf, np, axis = 1), osf, np, axis = 2)
gam = pynamr.downsample(pynamr.downsample(pynamr.downsample(gam, osf, np, axis = 0), osf, np, axis = 1), osf, np, axis = 2)
gam /= gam.max()

# generate the structual prior image
a_img = (x.max() - x)**0.5

# create pseudo-complex data
x = np.stack([x, 0*x], axis = -1)

# create sensitivity images
sens = np.ones((ncoils,n_ds,n_ds,n_ds)) + 0j*np.zeros((ncoils,n_ds,n_ds,n_ds))


#-------------------------------------------------------------------------------------
# generate a mono exp. acquisition model for 2 echoes
fwd_model = pynamr.MonoExpDualTESodiumAcqModel(data_shape, ds, ncoils, sens, dt, xp)

# generate data
y    = fwd_model.forward(x, gam)
data = y + noise_level*np.abs(y).mean()*np.random.randn(*y.shape).astype(np.float64)


#-------------------------------------------------------------------------------------
# setup the data fidelity loss function
data_fidelity_loss = pynamr.DataFidelityLoss(fwd_model, data)

aimg     = x[...,0]
aimg     = (aimg.max() - aimg)**0.5

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



nn_inds  = np.zeros((np.prod(aimg.shape), nnearest), dtype = np.uint32)
pynamr.nearest_neighbors(aimg, s, nnearest, nn_inds)
nn_inds_adj = pynamr.is_nearest_neighbor_of(nn_inds)   

bowsher_loss = pynamr.BowsherLoss(nn_inds, nn_inds_adj)

loss = pynamr.TotalLoss(data_fidelity_loss, bowsher_loss, bowsher_loss, beta_x, beta_gam)

#-------------------------------------------------------------------------------------
# run the recons

x_0   = 0*x + 1
gam_0 = 0*gam + 1

res_fix_1 = fmin_l_bfgs_b(loss.eval_x_first, x_0.ravel(), fprime = loss.grad_x, 
                          args = (gam_0.ravel(),), maxiter = n_inner, disp = 1)
x_fix = res_fix_1[0].reshape(x.shape)


res_fix_2 = fmin_l_bfgs_b(loss.eval_gam_first, gam_0.ravel(), fprime = loss.grad_gam, 
                          args = (x_0.ravel(),), maxiter = n_inner, disp = 1,
                          bounds = (gam_0.size)*[(0.001,1)])
gam_fix = res_fix_2[0].reshape(gam.shape)

x_r   = x_0.copy()
gam_r = gam_0.copy()

#------------------
# alternating LBFGS
for i_out in range(n_outer):
  res_1 = fmin_l_bfgs_b(loss.eval_x_first, x_r, fprime = loss.grad_x, 
                            args = (gam_r,), maxiter = n_inner, disp = 1)
  x_r = res_1[0].copy()

  res_2 = fmin_l_bfgs_b(loss.eval_gam_first, gam_r, fprime = loss.grad_gam, 
                            args = (x_r,), maxiter = n_inner, disp = 1,
                            bounds = (gam_0.size)*[(0.001,1)])

  gam_r = res_2[0].copy()

#------------------

x_r.reshape(x.shape)
gam_r.reshape(gam.shape)

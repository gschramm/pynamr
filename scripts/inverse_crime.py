import cupy as cp
import numpy as np
from scipy.ndimage import gaussian_filter

import pynamr

# input parameters
np.random.seed(1)
data_shape  = (64,64,64)
ds          = 2
ncoils      = 4
dt          = 5.
noise_level = 0
xp          = cp

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


################
################
q = fwd_model.adjoint(data, gam)

d = np.squeeze(data.astype(np.float64).view(dtype = np.complex128), axis = -1)
a = np.fft.ifftn(d[0,0,...], norm = 'ortho')

################
################

#from models import MonoExpDualTESodiumAcqModel
#from bowsher_prior import BowsherLoss, nearest_neighbors, is_nearest_neighbor_of
#
#
#n_outer     = 2
#n_inner     = 100
#
#data_shape  = (64,64,64)
#ds          = 2
#ncoils      = 3
#dt          = 5.
#noise_level = 0
#xp          = cp
#
#n_ds = data_shape[0] 
#n    = ds*n_ds
#
#tmp = np.pad(np.ones((n//2,n//2,n//2)), n//4)
#x = np.stack([np.random.randn(n,n,n),np.random.randn(n,n,n)], axis = -1)
#
##x = np.stack([tmp, tmp], axis = -1)
#
# 
#
#sens = np.random.rand(ncoils,n_ds,n_ds,n_ds) + 1j*np.random.rand(ncoils,n_ds,n_ds,n_ds)
##sens = np.ones((ncoils,n_ds,n_ds,n_ds)) + 1j*np.ones((ncoils,n_ds,n_ds,n_ds))
#sens *= 1e-2
#
#gam = np.random.rand(n,n,n)
##gam = np.ones((n,n,n))
##gam[tmp > 0] = 0.3
##gam[(n//2):,:,:] /= 2
##gam[tmp == 0] = 1
#
#fwd_model = MonoExpDualTESodiumAcqModel(data_shape, ds, ncoils, sens, dt, xp)
#
## generate data
#y = fwd_model.forward(x, gam)
#data = y + noise_level*np.abs(y).mean()*np.random.randn(*y.shape).astype(np.float64)
#
## setup data fidelity loss
#datafidelityloss = DataFidelityLoss(fwd_model, data)
#
#
## setup the Bowsher penalty loss
#prior_image = tmp + 0.01*np.random.randn(*tmp.shape)
#nnearest    = 4
#
#s   = np.array([[[0,1,0],[1,1,1],[0,1,0]], 
#                [[1,1,1],[1,0,1],[1,1,1]], 
#                [[0,1,0],[1,1,1],[0,1,0]]])
#
#nn_inds  = np.zeros((np.prod(prior_image.shape), nnearest), dtype = np.uint32)
#nearest_neighbors(prior_image, s, nnearest, nn_inds)
#nn_inds_adj = is_nearest_neighbor_of(nn_inds)   
#
#bowsher_penalty = BowsherLoss(nn_inds, nn_inds_adj)
#
#
## setup the combined loss function
#loss = TotalLoss(datafidelityloss, bowsher_penalty, bowsher_penalty, 1e-1, 1e-1)
#
#
## inital values
#x_0   = np.random.rand(*x.shape)
#gam_0 = np.random.rand(*gam.shape)
#
##x_0   = np.ones(x.shape)
##gam_0 = np.ones(gam.shape)
#
#
## check gradients
#ll = loss.eval_x_first(x_0, gam_0)
#gx = loss.grad_x(x_0, gam_0)
#gg = loss.grad_gam(gam_0, x_0)
#
#eps = 1e-6
#
#vox_nums = [40,51,63]
#
#for i in vox_nums:
#  delta_x = np.zeros(x.shape)
#  delta_x[i,i,i,0] = eps
#  print(gx[i,i,i,0], (loss.eval_x_first(x_0 + delta_x, gam_0) - ll) / eps)
#
#  delta_x = np.zeros(x.shape)
#  delta_x[i,i,i,1] = eps
#  print(gx[i,i,i,1], (loss.eval_x_first(x_0 + delta_x, gam_0) - ll) / eps)
#
#print('')
#
#for i in vox_nums:
#  delta_g = np.zeros(gam.shape)
#  delta_g[i,i,i] = eps
#  print(gg[i,i,i], (loss.eval_x_first(x_0, gam_0 + delta_g) - ll) / eps)
#
#
#
###----------------------------------------------------------------------------------------------
##x_0 = gaussian_filter(x, 2)
##
##gam_0 = np.ones((n,n,n))
##gam_0[tmp > 0] = 0.5
#
##x_r   = x_0.copy().ravel()
##gam_r = gam_0.copy().ravel()
#
##gam_bounds = (gam_0.size)*[(0.001,1)]
#
### LBFGS
##for i_out in range(n_outer):
##  res_1 = fmin_l_bfgs_b(loss.eval_x_first, x_r, fprime = loss.grad_x, 
##                            args = (gam_r,), maxiter = n_inner, disp = 1)
##  x_r = res_1[0].copy()
#
##  res_2 = fmin_l_bfgs_b(loss.eval_gam_first, gam_r, fprime = loss.grad_gam, 
##                            args = (x_r,), maxiter = n_inner, disp = 1,
##                            bounds = gam_bounds)
#
##  gam_r = res_2[0].copy()
#
#
##res_fix_1 = fmin_l_bfgs_b(loss.eval_x_first, x_0.ravel(), fprime = loss.grad_x, 
##                          args = (gam.ravel(),), maxiter = n_inner, disp = 1)
##x_fix = res_fix_1[0].reshape(x.shape)
#
##res_fix_2 = fmin_l_bfgs_b(loss.eval_gam_first, gam_0.ravel(), fprime = loss.grad_gam, 
##                          args = (x.ravel(),), maxiter = n_inner, disp = 1,
##                          bounds = (gam_0.size)*[(0.001,1)])
##gam_fix = res_fix_2[0].reshape(gam.shape)

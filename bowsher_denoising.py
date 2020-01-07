import numpy as np

from scipy.optimize    import fmin_bfgs
from nearest_neighbors import *
from bowsher           import *

#--------------------------------------------------------------------
def bowsher_denoising_cost(img, noisy_img, beta, ninds, ninds2, method):
  # ninds2 is a dummy argument to have the same arguments for
  # cost and its gradient

  data_fidelity = 0.5*((img - noisy_img)**2).sum()
  prior         = bowsher_prior_cost(img, ninds, method)

  cost = data_fidelity + beta*prior

  return cost

#--------------------------------------------------------------------
def bowsher_denoising_grad(img, noisy_img, beta, ninds, ninds2, method):

  data_fidelity_grad = img - noisy_img
  prior_grad         = bowsher_prior_grad(img, ninds, ninds2, method)

  grad = data_fidelity_grad + prior_grad

  return grad

#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------

np.random.seed(0)
nnearest = 2 
aimg     = np.pad(np.ones((20,20)),7,'constant')
aimg    += 0.01*np.random.random(aimg.shape)

if aimg.ndim == 3:
  s = np.array([[[0,1,0],[1,1,1],[0,1,0]], 
                [[1,1,1],[1,0,1],[1,1,1]], 
                [[0,1,0],[1,1,1],[0,1,0]]])

else:
  s = np.array([[0,1,0], 
                [1,0,1], 
                [0,1,0]])

ninds  = np.zeros((np.prod(aimg.shape),nnearest), dtype = np.uint32)
nearest_neighbors(aimg,s,nnearest,ninds)
ninds2 = is_nearest_neighbor_of(ninds)

#---

img        = 2*(1 - aimg) + 1
noisy_img  = img + 0.5*np.random.random(aimg.shape)
init_img   = noisy_img.copy()

##---
##---
#x0 = noisy_img.copy()
#x1 = x0.copy()
#tmp      = (16,16)
#eps      = 1e-5
#x1[tmp] += eps
#
#method = 1
#
#p0 = bowsher_prior_cost(x0, ninds, method)
#p1 = bowsher_prior_cost(x1, ninds, method)
#
#g  = bowsher_prior_grad(x0, ninds, ninds2, method)
#
##---
##---

method = 0
beta   = 1e0

res = fmin_bfgs(bowsher_denoising_cost, init_img.flatten(), fprime = bowsher_denoising_grad, 
                args = (noisy_img.flatten(), beta, ninds, ninds2, method), maxiter = 100,
                retall = True)

denoised_img = res[0].reshape(aimg.shape)

cost = [bowsher_denoising_cost(x, noisy_img.flatten(), beta, ninds, ninds2, method) for x in res[1]]

import matplotlib.pyplot as py
fig, ax = py.subplots(2,2,figsize=(7,7))
im_kwargs = {'vmin':0, 'vmax':noisy_img.max(), 'cmap':py.cm.Greys}
ax[0,0].imshow(img, **im_kwargs )
ax[0,1].imshow(aimg, **im_kwargs)
ax[1,0].imshow(noisy_img, **im_kwargs)
ax[1,1].imshow(denoised_img, **im_kwargs)
fig.tight_layout()
fig.show()

fig2, ax2 = py.subplots(1,1)
ax2.loglog(cost)
fig2.tight_layout()
fig2.show()

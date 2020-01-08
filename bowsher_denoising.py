import numpy as np

from scipy.optimize    import fmin_l_bfgs_b
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

# load the brain web labels
data  = np.load('54.npz')
gt    = data['arr_0'][...,65:75]
#gt    = data['arr_0'][...,75]
aimg  = (gt.max() - gt)**0.5
aimg += 0.001*aimg.max()*np.random.randn(*gt.shape)
img   = gt + 0.1*gt.max()*np.random.randn(*gt.shape)

# parameter for the bowsher prior
beta     = 1e0
nnearest = 3 
method   = 0

if aimg.ndim == 3:
  s = np.array([[[0,1,0],[1,1,1],[0,1,0]], 
                [[1,1,1],[1,0,1],[1,1,1]], 
                [[0,1,0],[1,1,1],[0,1,0]]])

else:
  s = np.array([[1,1,1], 
                [1,0,1], 
                [1,1,1]])

ninds  = np.zeros((np.prod(aimg.shape),nnearest), dtype = np.uint32)
nearest_neighbors(aimg,s,nnearest,ninds)
ninds2 = is_nearest_neighbor_of(ninds)

cost = []
cb   = lambda x: cost.append(bowsher_denoising_cost(x, img.flatten(), beta, ninds, ninds2, method))

bounds = [(0,None)] * len(img.flatten())

res = fmin_l_bfgs_b(bowsher_denoising_cost, 
                    img.flatten(), 
                    fprime = bowsher_denoising_grad, 
                    args = (img.flatten(), beta, ninds, ninds2, method),
                    bounds = bounds,
                    callback = cb,
                    disp = 1)

denoised_img = res[0].reshape(aimg.shape)

#--- show results

import matplotlib.pyplot as py
py.rcParams['image.cmap'] = 'Greys_r'
fig, ax = py.subplots(2,2,figsize=(7,7))
im_kwargs = {'vmin':0, 'vmax':1.3*gt.max(), 'cmap':py.cm.Greys_r}
if gt.ndim == 2:
  ax[0,0].imshow(gt.transpose(), **im_kwargs )
  ax[0,1].imshow(aimg.transpose())
  ax[1,0].imshow(img.transpose(), **im_kwargs)
  ax[1,1].imshow(denoised_img.transpose(), **im_kwargs)
else: 
  sl = gt.shape[-1]//2
  ax[0,0].imshow(gt[...,sl].transpose(), **im_kwargs )
  ax[0,1].imshow(aimg[...,sl].transpose())
  ax[1,0].imshow(img[...,sl].transpose(), **im_kwargs)
  ax[1,1].imshow(denoised_img[...,sl].transpose(), **im_kwargs)
fig.tight_layout()
fig.show()

fig2, ax2 = py.subplots(1,1)
ax2.loglog(np.arange(1,len(cost)+1),cost)
fig2.tight_layout()
fig2.show()

import numpy as np

from scipy.optimize    import fmin_l_bfgs_b
from scipy.ndimage     import gaussian_filter
from nearest_neighbors import *
from bowsher           import *

#--------------------------------------------------------------------
def bowsher_deblurring_cost(img, noisy_img, img_shape, sig, beta, ninds, ninds2, method):
  # ninds2 is a dummy argument to have the same arguments for
  # cost and its gradient

  isflat = False
  if img.ndim == 1:
    isflat    = True
    img       = img.reshape(img_shape)
    noisy_img = noisy_img.reshape(img_shape)

  cost = 0.5*((gaussian_filter(img, sig) - noisy_img)**2).sum()

  if beta > 0:
    cost += beta*bowsher_prior_cost(img, ninds, method)

  if isflat:
    img       = img.flatten()
    noisy_img = noisy_img.flatten()

  return cost

#--------------------------------------------------------------------
def bowsher_deblurring_grad(img, noisy_img, img_shape, sig, beta, ninds, ninds2, method):

  isflat = False
  if img.ndim == 1:
    isflat    = True
    img       = img.reshape(img_shape)
    noisy_img = noisy_img.reshape(img_shape)

  grad   = gaussian_filter(gaussian_filter(img, sig) - noisy_img, sig)

  if beta > 0:
    grad += beta*bowsher_prior_grad(img, ninds, ninds2, method)

  if isflat:
    img       = img.flatten()
    noisy_img = noisy_img.flatten()
    grad      = grad.flatten()

  return grad

#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------

np.random.seed(0)

sig         = 2.
noise_level = 0.05
beta        = 1e-1

# load the brain web labels
data  = np.load('54.npz')
gt    = data['arr_0'][...,65:75]
#gt    = data['arr_0'][...,75]
aimg  = (gt.max() - gt)**0.5
aimg += 0.001*aimg.max()*np.random.randn(*gt.shape)
img   = gaussian_filter(gt,sig) + noise_level*gt.max()*np.random.randn(*gt.shape)

img_shape = img.shape

# parameter for the bowsher prior
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
cb   = lambda x: cost.append(bowsher_deblurring_cost(x, img.flatten(), img_shape, sig, beta, ninds, ninds2, method))
#cb   = lambda x: cost.append(((x-gt.flatten())**2).sum())

bounds = [(None,None)] * len(img.flatten())

res = fmin_l_bfgs_b(bowsher_deblurring_cost, 
                    img.flatten(), 
                    fprime = bowsher_deblurring_grad, 
                    args = (img.flatten(), img_shape, sig, beta, ninds, ninds2, method),
                    bounds = bounds,
                    callback = cb,
                    maxiter = 200,
                    disp = 1)

denoised_img = res[0].reshape(aimg.shape)


#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
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
it = np.arange(1,len(cost)+1)
ax2.loglog(it, cost)
fig2.tight_layout()
fig2.show()

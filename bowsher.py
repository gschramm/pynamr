import numpy as np
from numba import njit
from nearest_neighbors import *

#--------------------------------------------------------------------
@njit()
def bowsher_prior_cost(img, ninds):
  img_shape = img.shape
  img       = img.flatten()
  cost      = 0.

  for i in range(ninds.shape[0]):
    # quadratic penalty
    cost += ((img[i] - img[ninds[i,:]])**2).sum()
  
  img = img.reshape(img_shape)

  return cost

#--------------------------------------------------------------------
@njit()
def bowsher_prior_grad(img, ninds, ninds2):
  img_shape = img.shape
  img       = img.flatten()
  grad      = np.zeros(img.shape, dtype = img.dtype)

  counter = 0

  for i in range(ninds.shape[0]):
    # first term
    # quadratic penalty
    grad[i] = 2*((img[i] - img[ninds[i,:]])).sum()
 
    # 2nd term
    while (counter < ninds2.shape[1]) and (ninds2[0,counter] == i):
      # quadratic penalty
      grad[i] +=  -2*(img[ninds2[1,counter]] - img[i])
      counter += 1

  img  = img.reshape(img_shape)
  grad = grad.reshape(img_shape)

  return grad

#--------------------------------------------------------------------

np.random.seed(0)
nnearest = 2 
aimg     = np.random.rand(50,50,50)

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

img  = np.random.random(aimg.shape)
img2 = img.copy()
eps  = 1e-5
img2[-1,-1,-1] += eps

p  = bowsher_prior_cost(img, ninds)
p2 = bowsher_prior_cost(img2, ninds)
g = bowsher_prior_grad(img, ninds, ninds2)

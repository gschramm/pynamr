import numpy as np
from numba import njit
from nearest_neighbors import *

@njit()
def bowsher_prior_cost(img, ninds):
  img_shape = img.shape
  img       = img.flatten()
  cost      = 0.

  for i in range(ninds.shape[0]):
    cost += ((img[ninds[i,:]] - img[i])**2).sum()
  
  img = img.reshape(img_shape)

  return cost

#--------------------------------------------------------------------

np.random.seed(0)
nnearest = 2
aimg     = np.random.rand(200,200,200)

s = np.array([[[0,1,0],[1,1,1],[0,1,0]], 
              [[1,1,1],[1,0,1],[1,1,1]], 
              [[0,1,0],[1,1,1],[0,1,0]]])

ninds  = np.zeros((np.prod(aimg.shape),nnearest), dtype = np.uint32)
nearest_neighbors(aimg,s,nnearest,ninds)
ninds2 = is_nearest_neighbor_of(ninds)

#---

img = np.random.random(aimg.shape)

p = bowsher_prior_cost(img, ninds)

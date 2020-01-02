#TODO: get rid of unravel / ravel_multi_index to allow njit

import numpy as np
from   numba import jit, njit

@njit()
def ravel_multi_index(multi_index, dims):

  flat_index = 0

  for i in range(len(multi_index)):
    offset = 1
    if i > 0:
      for j in range(i):
        offset *= dims[-(j+1)]

    flat_index += offset*multi_index[-(i+1)]

  return flat_index

#-------------------------------------------------------------------------------------

#@njit()
def nearest_neighbors(img, s, nnearest, ninds):
  ndim  = s.ndim
  sinds = np.where(s == 1)
  
  nmask  = sinds[0].shape[0] 
  offset = (np.array(s.shape) // 2)

  shape  = np.array(img.shape)

  for voxindex, voxvalue in np.ndenumerate(img):
    inds      = np.zeros(nmask, dtype = np.uint32)
    is_inside = np.ones(nmask, dtype = np.uint8)

    for j in range(nmask):
      tmp       = np.zeros(ndim, dtype = np.uint16)
    
      for i in range(ndim):
        tmp[i] = sinds[i][j] + voxindex[i] - offset[i]
    
        if tmp[i] < 0:  
          is_inside[j] = 0
        elif tmp[i] >= shape[i]: 
          is_inside[j] = 0
    
      if is_inside[j] == 1:
        inds[j] = ravel_multi_index(tmp, shape)

    inds = inds[is_inside == 1]
    # inds now contains the flattened indicies of the neighbors defined by the mask
    # around our given voxel

    # calculate absolute intensity difference to central voxel
    absdiff = np.abs(img[np.unravel_index(inds, shape)] - voxvalue)

    # contruct the nearest neighbor indices in the mask neighborhood
    ninds[np.ravel_multi_index(voxindex, shape),:] = inds[np.argsort(absdiff)][:nnearest]

#--------------------------------------------------------------------------------------

np.random.seed(0)
img = np.random.rand(4,4)
s   = np.array([[0,1,0],[1,0,1],[0,1,0]])
#s        = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,0,1,1],[0,1,1,1,0],[0,0,1,0,0]])
nnearest = 2

ninds = np.zeros((np.prod(img.shape), nnearest), dtype = np.uint32)
nearest_neighbors(img, s, nnearest, ninds)

import numpy as np
from numba import njit, prange

@njit(parallel = True)
def nearest_neighbors_3d(img,s,nnearest):
  offsets = np.array(s.shape) // 2
  maxdiff = img.max() - img.min() + 1 
  
  d12 = img.shape[1]*img.shape[2]
  d2  = img.shape[2]
  
  mask_center_offset = s.shape[0]*s.shape[1]*s.shape[2] // 2
  
  ninds = np.zeros((img.shape[0]*img.shape[1]*img.shape[2],nnearest), dtype = np.uint32)
  
  for i0 in prange(img.shape[0]):
    for i1 in range(img.shape[1]):
      for i2 in range(img.shape[2]):
  
        absdiff = np.zeros(s.shape)  
        val     = img[i0,i1,i2]
        
        i_flattened = np.zeros(s.shape, dtype = np.uint32)
  
        for j0 in range(s.shape[0]):
          for j1 in range(s.shape[1]):
            for j2 in range(s.shape[2]):
              tmp0 = i0 + j0 - offsets[0]
              tmp1 = i1 + j1 - offsets[1]
              tmp2 = i2 + j2 - offsets[2]
  
              i_flattened[j0,j1,j2] = tmp0*d12 + tmp1*d2 + tmp2
  
              if ((tmp0 >= 0) and (tmp0 < img.shape[0]) and 
                  (tmp1 >= 0) and (tmp1 < img.shape[1]) and 
                  (tmp2 >= 0) and (tmp2 < img.shape[2]) and s[j0,j1,j2] == 1):
                absdiff[j0,j1,j2] = np.abs(img[tmp0, tmp1, tmp2] - val)
              else:
                absdiff[j0,j1,j2] = maxdiff
 
        vox = i_flattened[offsets[0],offsets[1],offsets[2]]
        ninds[vox,:] = i_flattened.flatten()[np.argsort(absdiff.flatten())[:nnearest]]

  return ninds

#-------------------------------------------------------------------------------

np.random.seed(0)
img      = np.random.rand(2,3,4)
s        = np.array([[[0,1,0],[1,1,1],[0,1,0]], 
                     [[1,1,1],[1,0,1],[1,1,1]], 
                     [[0,1,0],[1,1,1],[0,1,0]]])
nnearest = 3

ninds = nearest_neighbors_3d(img,s,nnearest)

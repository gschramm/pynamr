import numpy as np
from   numba import jit, njit

def nearest_neighbors(img, s, nnearest):
  ndim  = s.ndim
  sinds = np.where(s == 1)
  
  nneighbors = sinds[0].shape[0] 
  offset     = (np.array(s.shape) // 2)

  ninds_1d = np.zeros((np.prod(img.shape), nnearest), dtype = np.uint32)

  for voxindex, voxvalue in np.ndenumerate(img):
    inds = []
    for j in range(nneighbors):
      tmp       = np.zeros(ndim, dtype = np.uint16)
      is_inside = True
    
      for i in range(ndim):
        tmp[i] = sinds[i][j] + voxindex[i] - offset[i]
    
        if tmp[i] < 0:  
          is_inside = False
        elif tmp[i] >= img.shape[i]: 
          is_inside = False
    
      if is_inside:
        inds.append(tmp)  
    
    # inds now contains the indicies of the neighbors defined by the mask
    # around our given voxel
    inds = tuple(np.array(inds).transpose().tolist())
    
    # calculate absolute intensity difference to central voxel
    absdiff = np.abs(img[inds] - voxvalue)
    
    # contruct the nearest neighbor indices in the mask neighborhood
    ninds = np.array(inds)[:,np.argsort(absdiff)[:nnearest]]
    ninds = tuple(ninds.tolist())
   
    # convert nearest neighbor inds to 1d flattened index
    ninds_1d[np.ravel_multi_index(voxindex, img.shape),:] = np.ravel_multi_index(ninds,img.shape)

  return ninds_1d

#--------------------------------------------------------------------------------------

np.random.seed(0)
img      = np.random.rand(200,200)
#s       = np.array([[0,1,0],[1,0,1],[0,1,0]])
s        = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,0,1,1],[0,1,1,1,0],[0,0,1,0,0]])
nnearest = 3

x = nearest_neighbors(img, s, nnearest)

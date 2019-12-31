# utils to implement the symmetric and asymmetric Bowsher prior

import numpy as np

img = np.arange(8*7).reshape(8,7)
#s   = np.array([[0,1,0],[1,0,1],[0,1,0]])
s = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,0,1,1],[0,1,1,1,0],[0,0,1,0,0]])

ndim  = s.ndim
sinds = np.where(s == 1)

nneighbors = sinds[0].shape[0] 

vox    = [7,6]
offset = (np.array(s.shape) // 2)

inds = []
for j in range(nneighbors):
  tmp       = np.zeros(ndim, dtype = np.int64)
  is_inside = True

  for i in range(ndim):
    tmp[i] = sinds[i][j] + vox[i] - offset[i]

    if tmp[i] < 0:  
      is_inside = False
    elif tmp[i] >= img.shape[i]: 
      is_inside = False

  if is_inside:
    inds.append(tmp)  

inds = tuple(np.array(inds).transpose().tolist())

print(img)
print(img[vox[0],vox[1]])
print(img[inds])

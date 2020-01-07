import math
from   numba import njit, prange

#--------------------------------------------------------------
@njit(parallel = True)
def prox_tv(x, beta):
  for i in prange(x.shape[1]):
    for j in range(x.shape[2]):
      norm = 0
      for k in range(x.shape[0]):
        norm += x[k,i,j]**2 

      norm = math.sqrt(norm) / beta

      if norm > 1:
        for k in range(x.shape[0]):
          x[k,i,j] /= norm

#--------------------------------------------------------------
@njit(parallel = True)
def prox_pls(x, xi, beta, method):
  for i in prange(x.shape[1]):
    for j in range(x.shape[2]):

      # calculate the norm of the joint vector field
      norm_xi = (xi[:,i,j]**2).sum()

      if norm_xi > 0:
        sp     = (x[:,i,j] * xi[:,i,j]).sum() / norm_xi
        x_perp = x[:,i,j] - sp*xi[:,i,j]
      else: 
        x_perp = x[:,i,j].copy()
     
      norm_ratio = (x_perp*x_perp).sum()

      if method == 1:
        if norm_xi > 0:
          norm_ratio = math.sqrt(norm_ratio) / (beta * math.sqrt(norm_xi) )
          if norm_ratio > 1:
            x[:,i,j] = x_perp / norm_ratio
          else:
            x[:,i,j] = x_perp
        else: 
          x[:,i,j] = 0

      elif method == 2:
        # PLS2
        norm_ratio = math.sqrt(norm_ratio) / beta

        if norm_ratio > 1:
          x[:,i,j] = x_perp / norm_ratio
        else:
          x[:,i,j] = x_perp

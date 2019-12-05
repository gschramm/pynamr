# small demo script to verify implementation of discrete FT (with FFT)

import numpy as np
from   numba import njit, prange

import matplotlib.pyplot as py
from   matplotlib.colors import LogNorm
py.ion()

#--------------------------------------------------------------
# this is the super naiv direct implementation of the DFT is 
# horribly slow. For the apodized FFT is more efficient
# to calculate multiple FFTs
@njit(parallel = True)
def dft2D(f, k0, k1):
  n0,n1 = f.shape
  F     = np.zeros((n0,n1), dtype = np.complex64)
  x0    = np.outer(np.arange(n0),np.ones(n1))
  x1    = np.outer(np.ones(n0),np.arange(n1))

  for i in prange(n0): 
    for j in range(n1): 
      F[i,j] = (f*np.exp(-2*np.pi*(k0[i]*x0 + k1[j]*x1)*1j)).sum()

  return F

#--------------------------------------------------------------

n = 256
x = np.arange(n)
f = np.random.rand(n,n) + np.random.rand(n,n)*1j
k = np.fft.fftfreq(n)

F1 = np.fft.fft2(f)
F2 = dft2D(f,k,k)

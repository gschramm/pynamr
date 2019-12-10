# script to test adjointness of imaginary part restriction opertator
# x ... complex vector Re(x) + iIm(x)
# y ... real vector 
# Im x  = Im(x)
# Im* y = 0 + i*y 
# inner product Re(x*y)

import numpy as np

def Im(x):
  return x.imag

def Im_adjoint(y):
  tmp = np.zeros(y.shape, dtype = np.complex)
  tmp.imag = y

  return tmp

#----------------------------------------------

n = 16

x      = np.zeros((n,n,n), dtype = np.complex)
x.real = np.random.random(x.shape)
x.imag = np.random.random(x.shape)

y = np.random.random(x.shape)

x_fwd  = Im(x)
y_back = Im_adjoint(y)


print((y.conj()*x_fwd).real.sum())
print((y_back.conj()*x).real.sum())

import numpy as np
from grad import *

n    = 16
ndim = 4

x = np.pad(np.random.random(ndim*(n-4,)) + np.random.random(ndim*(n-4,))*1j, (2,2), 'constant')

x_fwd = np.zeros((2*x.ndim,) + x.shape)
complex_grad(x,x_fwd)

y      = np.random.random((2*x.ndim,) + x.shape)
y_back = complex_div(y)  

print((x.real*y_back.real).sum())
print((x_fwd[:ndim,...]*y[:ndim,...]).sum())

print((x.imag*y_back.imag).sum())
print((x_fwd[ndim:,...]*y[ndim:,...]).sum())

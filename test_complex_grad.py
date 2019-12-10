import numpy as np
import pymirc.image_operations as pymi

n    = 256
ndim = 2

x = np.pad(np.random.random(ndim*(n-4,)) + np.random.random(ndim*(n-4,))*1j, (2,2), 'constant')

x_fwd = np.zeros((2*x.ndim,) + x.shape)
pymi.complex_grad(x,x_fwd)

y      = np.random.random((2*x.ndim,) + x.shape)
y_back = -pymi.complex_div(y)  

print((y.conj()*x_fwd).sum().real)
print((y_back.conj()*x).sum().real)

print("")

# power iterations
b = x.copy()
for it in range(200):
  tmp = np.zeros((2*ndim,) + ndim*(n,))
  pymi.complex_grad(b, tmp)
  b_fwd = -pymi.complex_div(tmp)
  L     = np.sqrt((b_fwd.real**2 + b_fwd.imag**2).sum())
  b     = b_fwd / L
  print(L)

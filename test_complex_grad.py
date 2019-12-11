import numpy as np
import pymirc.image_operations as pymi

n    = 256
ndim = 2

x = np.pad(np.random.random(ndim*(n-4,) + (2,)), ndim*[(2,2)] + [(0,0)], 'constant')

x_fwd = np.zeros((2*ndim,) + x.shape[:-1])
pymi.complex_grad(x,x_fwd)

y      = np.random.random(x_fwd.shape)
y_back = -pymi.complex_div(y)  

print((x_fwd*y).sum())
print((x*y_back).sum())

print("")

# power iterations
b = x.copy()
for it in range(100000):
  tmp = np.zeros(x_fwd.shape)
  pymi.complex_grad(b, tmp)
  b_fwd = -pymi.complex_div(tmp)
  L     = np.sqrt((b_fwd*b_fwd).sum())
  b     = b_fwd / L
  print(L)

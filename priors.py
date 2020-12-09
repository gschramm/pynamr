import numpy as np
from pymirc.image_operations import grad, div

def quadratic_prior(x):
  g = np.zeros((x.ndim,) + x.shape, dtype = x.dtype)
  grad(x,g)

  return 0.5*(g**2).sum() 

def quadratic_prior_grad(x):
  g = np.zeros((x.ndim,) + x.shape, dtype = x.dtype)
  grad(x,g)

  return -div(g)

def logcosh_prior(x, delta = 10):
  g = np.zeros((x.ndim,) + x.shape, dtype = x.dtype)
  grad(x,g)

  return np.log(np.cosh(delta*g)).sum() / delta

def logcosh_prior_grad(x, delta = 10):
  g = np.zeros((x.ndim,) + x.shape, dtype = x.dtype)
  grad(x,g)

  return -div(np.tanh(delta*g))

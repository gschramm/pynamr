#TODO: f_min_l_bfgs seems to only work with 64bit float 1D arrays -> problem with complex64 data

import numpy as np
from scipy.optimize import fmin_l_bfgs_b, fmin_cg

#-------------------------------------------------------------------

class DataFidelityLoss:
  """ Data fidelity loss for mono exponential dual echo sodium forward model
     
      The model is linear in the first argument (the image x), but non linear
      in the second argument (Gam)
  """

  def __init__(self, model, y):
    self.model = model
    self.y     = y
 
  def eval_x_first(self, x, gam):
    z = self.diff(x, gam)
    return 0.5*(z**2).sum()

  def eval_gam_first(self, gam, x):
    return self.eval_x_first(x, gam)

  def diff(self, x, gam):
    # reshaping of x/gam is needed since fmin_l_bfgs_b flattens all arrays
    return (self.model.forward(x.reshape(self.model._image_shape + (2,)) , 
                               gam.reshape(self.model._image_shape)) - self.y) * self.model._kmask

  def grad_x(self, x, gam):
    in_shape = x.shape
    z = self.diff(x, gam)

    # reshaping of x/gam is needed since fmin_l_bfgs_b flattens all arrays
    return self.model.adjoint(z, gam.reshape(self.model._image_shape)).reshape(in_shape)

  def grad_gam(self, gam, x):
    in_shape = gam.shape
    z = self.diff(x, gam)

    # reshaping of x/gam is needed since fmin_l_bfgs_b flattens all arrays
    return self.model.grad_gam(z, gam.reshape(self.model._image_shape), x.reshape(self.model._image_shape + (2,))).reshape(in_shape)



#-------------------------------------------------------------------

if __name__ == '__main__':

  import cupy as cp
  from scipy.ndimage import gaussian_filter
  from models import MonoExpDualTESodiumAcqModel

  np.random.seed(1)

  data_shape  = (64,64,64)
  ds          = 2
  ncoils      = 3
  dt          = 5.
  noise_level = 0
  n_inner     = 20
  xp          = cp

  n_ds = data_shape[0] 
  n    = ds*n_ds

  tmp = gaussian_filter(np.pad(np.ones((n//2,n//2,n//2), dtype = np.float64), n//4),2)
  x   = np.stack([tmp,tmp], axis = -1)

  sens = 1e-2*(np.ones((ncoils,n_ds,n_ds,n_ds)).astype(np.float64) + 1j*np.ones((ncoils,n_ds,n_ds,n_ds)).astype(np.float64))
 
  gam = 0.9*np.ones((n,n,n)).astype(np.float64)
  
  fwd_model = MonoExpDualTESodiumAcqModel(data_shape, ds, ncoils, sens, dt, xp)
  
  # generate data
  y = fwd_model.forward(x, gam)
  data = y + noise_level*np.abs(y).mean()*np.random.randn(*y.shape).astype(np.float64)

  # setup data fidelity loss
  loss = DataFidelityLoss(fwd_model, data)

  # inital values
  x_0   = 1.1*x
  gam_0 = 0.7*gam

  # check gradients
  ll = loss.eval_x_first(x_0, gam_0)
  gx = loss.grad_x(x_0, gam_0)
  gg = loss.grad_gam(gam_0, x_0)

  eps = 1e-6

  delta_x = np.zeros(x.shape)
  delta_x[64,64,64,0] = eps
  print(gx[64,64,64,0], (loss.eval_x_first(x_0 + delta_x, gam_0) - ll) / eps)


  for i in [40,51,63]:
    delta_g = np.zeros(gam.shape)
    delta_g[i,i,i] = eps
    print(gg[i,i,i], (loss.eval_x_first(x_0, gam_0 + delta_g) - ll) / eps)




  ##----------------------------------------------------------------------------------------------

  #x_r   = x_0.copy()
  #gam_r = gam_0.copy()

  ## LBFGS
  ##for i_out in range(n_outer):
  ##  res = fmin_l_bfgs_b(loss.eval_x_first, x_r, fprime = loss.grad_x, args = (gam_r,), 
  ##                      maxiter = n_inner, disp = 1)
  ##  x_r = res[0].copy()

  ##  res_gam = fmin_l_bfgs_b(loss.eval_gam_first, gam_r, fprime = loss.grad_gam, args = (x_r,), 
  ##                          maxiter = n_inner, disp = 1)
  ##  gam_r = res_gam[0].copy()
  #
  #res_fix_1 = fmin_l_bfgs_b(loss.eval_x_first, x_0.ravel().astype(np.float64), 
  #                          fprime = loss.grad_x, args = (gam.ravel().astype(np.float64),), 
  #                          maxiter = n_inner, disp = 1)
  #res_fix_2 = fmin_l_bfgs_b(loss.eval_gam_first, gam_0, fprime = loss.grad_gam, args = (x,), 
  #                          maxiter = n_inner, disp = 0)
  #res_fix_3 = fmin_l_bfgs_b(loss.eval_gam_first, 1.2*gam, fprime = loss.grad_gam, args = (x,), 
  #                          maxiter = n_inner, disp = 0)

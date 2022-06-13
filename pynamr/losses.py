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
    return self.model.grad_gam(z, gam.reshape(self.model._image_shape), x.reshape(self.model._image_shape + (2,))).reshape(in_shape).copy()



#-------------------------------------------------------------------

if __name__ == '__main__':

  import cupy as cp
  from scipy.ndimage import gaussian_filter
  from models import MonoExpDualTESodiumAcqModel

  np.random.seed(1)

  n_outer     = 2
  n_inner     = 100

  data_shape  = (64,64,64)
  ds          = 2
  ncoils      = 3
  dt          = 5.
  noise_level = 0
  xp          = cp

  n_ds = data_shape[0] 
  n    = ds*n_ds

  #x = np.stack([np.random.randn(n,n,n),np.random.randn(n,n,n)], axis = -1)

  tmp = np.pad(np.ones((n//2,n//2,n//2)), n//4)
  x = np.stack([tmp,tmp], axis = -1)

  #sens = np.random.rand(ncoils,n_ds,n_ds,n_ds) + 1j*np.random.rand(ncoils,n_ds,n_ds,n_ds)
  sens = np.ones((ncoils,n_ds,n_ds,n_ds)) + 1j*np.ones((ncoils,n_ds,n_ds,n_ds))
  sens *= 1e-2

  #gam = np.random.rand(n,n,n)
  gam = np.ones((n,n,n))
  gam[tmp > 0] = 0.3
  gam[(n//2):,:,:] /= 2
  gam[tmp == 0] = 1
  
  fwd_model = MonoExpDualTESodiumAcqModel(data_shape, ds, ncoils, sens, dt, xp)
  
  # generate data
  y = fwd_model.forward(x, gam)
  data = y + noise_level*np.abs(y).mean()*np.random.randn(*y.shape).astype(np.float64)

  # setup data fidelity loss
  loss = DataFidelityLoss(fwd_model, data)

  # inital values
  #x_0   = np.random.rand(*x.shape)
  #gam_0 = np.random.rand(*gam.shape)

  x_0   = np.ones(x.shape)
  gam_0 = np.ones(gam.shape)


  ## check gradients
  #ll = loss.eval_x_first(x_0, gam_0)
  #gx = loss.grad_x(x_0, gam_0)
  #gg = loss.grad_gam(gam_0, x_0)

  #eps = 1e-6

  #vox_nums = [40,51,63]

  #for i in vox_nums:
  #  delta_x = np.zeros(x.shape)
  #  delta_x[i,i,i,0] = eps
  #  print(gx[i,i,i,0], (loss.eval_x_first(x_0 + delta_x, gam_0) - ll) / eps)

  #  delta_x = np.zeros(x.shape)
  #  delta_x[i,i,i,1] = eps
  #  print(gx[i,i,i,1], (loss.eval_x_first(x_0 + delta_x, gam_0) - ll) / eps)

  #print('')

  #for i in vox_nums:
  #  delta_g = np.zeros(gam.shape)
  #  delta_g[i,i,i] = eps
  #  print(gg[i,i,i], (loss.eval_x_first(x_0, gam_0 + delta_g) - ll) / eps)



  #----------------------------------------------------------------------------------------------
  x_0 = gaussian_filter(x, 2)
  
  gam_0 = np.ones((n,n,n))
  gam_0[tmp > 0] = 0.5

  x_r   = x_0.copy().ravel()
  gam_r = gam_0.copy().ravel()

  gam_bounds = (gam_0.size)*[(0.001,1)]

  # LBFGS
  for i_out in range(n_outer):
    res_1 = fmin_l_bfgs_b(loss.eval_x_first, x_r, fprime = loss.grad_x, 
                              args = (gam_r,), maxiter = n_inner, disp = 1)
    x_r = res_1[0].copy()

    res_2 = fmin_l_bfgs_b(loss.eval_gam_first, gam_r, fprime = loss.grad_gam, 
                              args = (x_r,), maxiter = n_inner, disp = 1,
                              bounds = gam_bounds)
 
    gam_r = res_2[0].copy()

 
  #res_fix_1 = fmin_l_bfgs_b(loss.eval_x_first, x_0.ravel(), fprime = loss.grad_x, 
  #                          args = (gam.ravel(),), maxiter = n_inner, disp = 1)
  #x_fix = res_fix_1[0].reshape(x.shape)

  #res_fix_2 = fmin_l_bfgs_b(loss.eval_gam_first, gam_0.ravel(), fprime = loss.grad_gam, 
  #                          args = (x.ravel(),), maxiter = n_inner, disp = 1,
  #                          bounds = (gam_0.size)*[(0.001,1)])
  #gam_fix = res_fix_2[0].reshape(gam.shape)

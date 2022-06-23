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

class TotalLoss:
  def __init__(self, datafidelityloss, penalty_x, penalty_gam, beta_x, beta_gam):
    self.datafidelityloss = datafidelityloss
    self.penalty_x        = penalty_x
    self.penalty_gam      = penalty_gam

    self.beta_x   = beta_x
    self.beta_gam = beta_gam

  def eval_x_first(self, x, gam):
    cost = datafidelityloss.eval_x_first(x, gam)

    if self.beta_x > 0:
      cost += self.beta_x * self.penalty_x.eval(x[...,0])
      cost += self.beta_x * self.penalty_x.eval(x[...,1])

    if self.beta_gam > 0:
      cost += self.beta_gam * self.penalty_gam.eval(gam)

    return cost

  def eval_gam_first(self, gam, x):
    return self.eval_x_first(x, gam)

  def grad_x(self, x, gam):
    grad = self.datafidelityloss.grad_x(x, gam)

    if self.beta_x > 0:
      grad[...,0] += self.beta_x * self.penalty_x.grad(x[...,0])
      grad[...,1] += self.beta_x * self.penalty_x.grad(x[...,1])

    return grad

  def grad_gam(self, gam, x):
    grad = self.datafidelityloss.grad_gam(gam, x)

    if self.beta_gam > 0:
      grad += self.beta_gam * self.penalty_gam.grad(gam)

    return grad

#-------------------------------------------------------------------

if __name__ == '__main__':

  import cupy as cp
  from scipy.ndimage import gaussian_filter
  from models import MonoExpDualTESodiumAcqModel
  from bowsher_prior import BowsherLoss, nearest_neighbors, is_nearest_neighbor_of

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

  tmp = np.pad(np.ones((n//2,n//2,n//2)), n//4)
  x = np.stack([np.random.randn(n,n,n),np.random.randn(n,n,n)], axis = -1)

  sens = np.random.rand(ncoils,n_ds,n_ds,n_ds) + 1j*np.random.rand(ncoils,n_ds,n_ds,n_ds)
  sens *= 1e-2

  gam = np.random.rand(n,n,n)
  
  fwd_model = MonoExpDualTESodiumAcqModel(data_shape, ds, ncoils, sens, dt, xp)
  
  # generate data
  y = fwd_model.forward(x, gam)
  data = y + noise_level*np.abs(y).mean()*np.random.randn(*y.shape).astype(np.float64)

  # setup data fidelity loss
  datafidelityloss = DataFidelityLoss(fwd_model, data)


  # setup the Bowsher penalty loss
  prior_image = tmp + 0.01*np.random.randn(*tmp.shape)
  nnearest    = 4

  s   = np.array([[[0,1,0],[1,1,1],[0,1,0]], 
                  [[1,1,1],[1,0,1],[1,1,1]], 
                  [[0,1,0],[1,1,1],[0,1,0]]])
  
  nn_inds  = np.zeros((np.prod(prior_image.shape), nnearest), dtype = np.uint32)
  nearest_neighbors(prior_image, s, nnearest, nn_inds)
  nn_inds_adj = is_nearest_neighbor_of(nn_inds)   
 
  bowsher_penalty = BowsherLoss(nn_inds, nn_inds_adj)


  # setup the combined loss function
  loss = TotalLoss(datafidelityloss, bowsher_penalty, bowsher_penalty, 1e-2, 1e-2)


  # inital values
  x_0   = np.random.rand(*x.shape)
  gam_0 = np.random.rand(*gam.shape)

  # check gradients
  ll = loss.eval_x_first(x_0, gam_0)
  gx = loss.grad_x(x_0, gam_0)
  gg = loss.grad_gam(gam_0, x_0)

  eps = 1e-5

  vox_nums = [40,51,63]

  for i in vox_nums:
    delta_x = np.zeros(x.shape)
    delta_x[i,i,i,0] = eps
    print(gx[i,i,i,0], (loss.eval_x_first(x_0 + delta_x, gam_0) - ll) / eps)

    delta_x = np.zeros(x.shape)
    delta_x[i,i,i,1] = eps
    print(gx[i,i,i,1], (loss.eval_x_first(x_0 + delta_x, gam_0) - ll) / eps)

  print('')

  for i in vox_nums:
    delta_g = np.zeros(gam.shape)
    delta_g[i,i,i] = eps
    print(gg[i,i,i], (loss.eval_x_first(x_0, gam_0 + delta_g) - ll) / eps)

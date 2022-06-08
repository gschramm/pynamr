import numpy as np
from scipy.optimize import fmin_l_bfgs_b, fmin_cg

class FwdModel:
  def __init__(self, A):
    self.A  = A
    self.ny, self.nx = self.A.shape
    self.weights = np.eye(self.ny).tolist()

  def __call__(self, x, gam):
    cost = 0

    for i in range(self.ny):
      cost += self.weights[i] * (self.A @ ((gam**i) * x))
 
    return cost

  def grad_x(self, x, gam, outer_deriv = 1):
    grad = np.zeros(self.nx)

    for i in range(self.ny):
      grad += (gam**i) * (self.A.T @ (self.weights[i] * outer_deriv))

    return grad

  def grad_gam(self, x, gam, outer_deriv = 1):
    grad = np.zeros(self.nx)

    for i in range(self.ny):
      grad += i * (gam**(i-1)) * x * (self.A.T @ (self.weights[i] * outer_deriv))

    return grad


#-------------------------------------------------------------------

class DataFidelityLoss:
  def __init__(self, model, y):
    self.model = model
    self.y     = y
 
  def eval_x_first(self, x, gam):
    z = self.diff(x, gam)
    return 0.5*(z**2).sum()

  def outer_deriv(self, x):
    return x

  def eval_gam_first(self, gam, x):
    return self.eval_x_first(x, gam)

  def diff(self, x, gam):
    return (self.model(x, gam) - self.y)

  def grad_x(self, x, gam):
    z = self.diff(x, gam)
    return self.model.grad_x(x, gam, self.outer_deriv(z))

  def grad_gam(self, gam, x):
    z = self.diff(x, gam)
    return self.model.grad_gam(x, gam, self.outer_deriv(z))



#-------------------------------------------------------------------

if __name__ == '__main__':
  np.random.seed(1)
 
  n_outer = 100
  n_inner = 50
  nl      = 0
 
  x   = np.array([1.5,0.5,2.5,0.8])
  gam = np.array([2.3,4.7,1.2,3.2])
 
  # setup forward model 
  A   = np.random.rand(5,4)
  fwd_model = FwdModel(A) 

  # generate data
  y =  fwd_model(x, gam)
  data = y + nl*np.random.normal(size = A.shape[0])

  loss = DataFidelityLoss(fwd_model, data)

  # inital values
  x_0   = np.random.rand(A.shape[1])
  gam_0 = np.random.rand(A.shape[1])

  # test the gradient
  ll = loss.eval_x_first(x_0, gam_0)
  gx = loss.grad_x(x_0,gam_0)
  gg = loss.grad_gam(gam_0,x_0)

  eps = 1e-6

  for i in range(A.shape[1]):
    delta = np.zeros(A.shape[1])
    delta[i] = eps  

    print(gx[i], (loss.eval_x_first(x_0 + delta, gam_0) - ll)  / eps)
    print(gg[i], (loss.eval_gam_first(gam_0 + delta, x_0) - ll)  / eps)
    print('')

  #----------------------------------------------------------------------------------------------

  x_r   = x_0.copy()
  gam_r = gam_0.copy()

  # LBFGS
  #for i_out in range(n_outer):
  #  res = fmin_l_bfgs_b(loss.eval_x_first, x_r, fprime = loss.grad_x, args = (gam_r,), 
  #                      maxiter = n_inner, disp = 1)
  #  x_r = res[0].copy()

  #  res_gam = fmin_l_bfgs_b(loss.eval_gam_first, gam_r, fprime = loss.grad_gam, args = (x_r,), 
  #                          maxiter = n_inner, disp = 1)
  #  gam_r = res_gam[0].copy()

  res_fix_1 = fmin_l_bfgs_b(loss.eval_x_first, x_0, fprime = loss.grad_x, args = (gam,), 
                            maxiter = n_inner, disp = 0)
  res_fix_2 = fmin_l_bfgs_b(loss.eval_gam_first, gam_0, fprime = loss.grad_gam, args = (x,), 
                            maxiter = n_inner, disp = 0)
  res_fix_3 = fmin_l_bfgs_b(loss.eval_gam_first, 1.2*gam, fprime = loss.grad_gam, args = (x,), 
                            maxiter = n_inner, disp = 0)

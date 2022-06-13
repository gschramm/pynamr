import unittest
import numpy as np
import pynamr

class TestGradients(unittest.TestCase):

  def setUp(self):
    self.data_shape  = (32,32,32)
    self.ds          = 2
    self.ncoils      = 3
    self.dt          = 5.
    self.noise_level = 0.1
    self.xp          = np
   
    #---------------------------------------------------------------------- 
    np.random.seed(1)

    self.n_ds = self.data_shape[0] 
    self.n    = self.ds*self.n_ds
    self.img_shape = (self.n,self.n,self.n)    

    self.x = np.stack([np.random.randn(*self.img_shape),np.random.randn(*self.img_shape)], axis = -1)
    
    self.sens = np.random.rand(*((self.ncoils,) + self.data_shape)) + 1j*np.random.rand(*((self.ncoils,) + self.data_shape))
    self.sens *= 1e-2
    
    self.gam = np.random.rand(*self.img_shape)
    
    self.fwd_model = pynamr.MonoExpDualTESodiumAcqModel(self.data_shape, self.ds, self.ncoils, 
                                                        self.sens, self.dt, self.xp)
    
    # generate data
    self.y = self.fwd_model.forward(self.x, self.gam)
    self.data = self.y + self.noise_level*np.abs(self.y).mean()*np.random.randn(*self.y.shape)
    

  def test_data_fidelity_gradients(self, i = 51, eps  = 1e-4, rtol = 1e-3):
    # setup data fidelity loss
    loss = pynamr.DataFidelityLoss(self.fwd_model, self.data)

    # inital values
    x_0   = np.random.rand(*self.x.shape)
    gam_0 = np.random.rand(*self.gam.shape)
    
    # check gradients
    ll = loss.eval_x_first(x_0, gam_0)
    gx = loss.grad_x(x_0, gam_0)
    gg = loss.grad_gam(gam_0, x_0)
    
    # test gradient with respect to Na image (real part) 
    delta_x = np.zeros(x_0.shape)
    delta_x[i,i,i,0] = eps
    self.assertTrue(np.isclose(gx[i,i,i,0], (loss.eval_x_first(x_0 + delta_x, gam_0) - ll) / eps, 
                               rtol = rtol))
    
    # test gradient with respect to Na image (imag part) 
    delta_x = np.zeros(x_0.shape)
    delta_x[i,i,i,1] = eps
    self.assertTrue(np.isclose(gx[i,i,i,1], (loss.eval_x_first(x_0 + delta_x, gam_0) - ll) / eps, 
                               rtol = rtol))
    
    # test gradient with respect to Gamma image 
    delta_g = np.zeros(gam_0.shape)
    delta_g[i,i,i] = eps
    self.assertTrue(np.isclose(gg[i,i,i], (loss.eval_x_first(x_0, gam_0 + delta_g) - ll) / eps, 
                               rtol = rtol))

#---------------------------------------------------------------------------------------------------

if __name__ == '__main__':
  unittest.main()

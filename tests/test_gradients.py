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
    

  def test_data_fidelity_gradient(self, i = 51, eps  = 1e-4, rtol = 1e-3):
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

  def test_total_gradient(self, i = 51, eps  = 1e-4, rtol = 1e-3, beta_x = 1e-2, beta_gam = 1e-2):
    # setup data fidelity loss
    data_fidelity_loss = pynamr.DataFidelityLoss(self.fwd_model, self.data)

    # setup the Bowsher loss
    nnearest = 2
    aimg     = np.random.rand(*self.img_shape)

    s   = np.array([[[0,1,0],[1,1,1],[0,1,0]], 
                    [[1,1,1],[1,0,1],[1,1,1]], 
                    [[0,1,0],[1,1,1],[0,1,0]]])
    
    nn_inds  = np.zeros((np.prod(self.img_shape), nnearest), dtype = np.uint32)
    pynamr.nearest_neighbors(aimg, s, nnearest, nn_inds)
    nn_inds_adj = pynamr.is_nearest_neighbor_of(nn_inds)   
 
    bowsher_loss = pynamr.BowsherLoss(nn_inds, nn_inds_adj)

    # setup the combined loss function
    loss = pynamr.TotalLoss(data_fidelity_loss, bowsher_loss, bowsher_loss, beta_x, beta_gam)

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



  def test_bowsher_gradient(self, eps  = 1e-7, rtol = 1e-3, atol = 1e-4):
    nnearest  = 2
    img_shape = (4,5,6)
    aimg      = np.random.rand(*img_shape)
    timg      = np.random.rand(*img_shape)
    
    s   = np.array([[[0,1,0],[1,1,1],[0,1,0]], 
                    [[1,1,1],[1,0,1],[1,1,1]], 
                    [[0,1,0],[1,1,1],[0,1,0]]])
    
    nn_inds  = np.zeros((np.prod(img_shape), nnearest), dtype = np.uint32)
    pynamr.nearest_neighbors(aimg, s, nnearest, nn_inds)
    nn_inds_adj = pynamr.is_nearest_neighbor_of(nn_inds)   
 
    bl = pynamr.BowsherLoss(nn_inds, nn_inds_adj)

    c1 = bl.eval(timg)
 
    g1 = bl.grad(timg)

    # check gradient numerically 
    close = np.zeros(img_shape, dtype = np.uint8)

    for i in range(img_shape[0]):
      for j in range(img_shape[1]):
        for k in range(img_shape[2]):
          delta = np.zeros(img_shape)
          delta[i,j,k] = eps

          g2 = (bl.eval(timg + delta) - c1) / eps

          close[i,j,k] = np.isclose(g1[i,j,k], g2, rtol = rtol, atol = atol)

    self.assertTrue(np.all(close))
#---------------------------------------------------------------------------------------------------

if __name__ == '__main__':
  unittest.main()

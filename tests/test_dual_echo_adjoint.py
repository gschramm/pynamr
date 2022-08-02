import unittest
import numpy as np
import pynamr

class TestModelAdjoint(unittest.TestCase):
  """ test wheter adjoint of signal model for mono exp. dual echo sodium signal model
      is correct 
  """
  def test_model_adjoint(self):
    np.random.seed(1)
    
    data_shape = (32,32,32)
    ds     = 2
    ncoils = 4
    dt     = 5.
    xp     = np
    
    n_ds = data_shape[0] 
    n    = ds*n_ds
    
    a = np.pad(np.random.rand(n-4,n-4,n-4),2).astype(np.float64)
    b = np.pad(np.random.rand(n-4,n-4,n-4),2).astype(np.float64)
    f = np.stack([a,b], axis = -1)
    
    sens = np.random.rand(ncoils,n_ds,n_ds,n_ds).astype(np.float64) + 1j*np.random.rand(ncoils,n_ds,n_ds,n_ds).astype(np.float64)
    
    Gam = np.random.rand(n,n,n).astype(np.float64)
    
    m = pynamr.MonoExpDualTESodiumAcqModel(data_shape, ds, ncoils, sens, dt, xp)
    
    f_fwd  = m.forward(f, Gam)
    F      = np.random.rand(*f_fwd.shape).astype(np.float64)
    F_back = m.adjoint(F, Gam) 
    
    self.assertTrue(np.isclose((f_fwd*F).sum(), (f*F_back).sum()))

#---------------------------------------------------------------------------------------------------

if __name__ == '__main__':
  unittest.main()

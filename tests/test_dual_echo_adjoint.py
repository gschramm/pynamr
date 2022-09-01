import re
import unittest
import numpy as np
import pynamr


class TestModelAdjoint(unittest.TestCase):
    """ test wheter adjoint of signal model for mono exp. dual echo sodium signal model
      is correct 
  """

    def setUp(self) -> None:
        self.data_shape = (32, 32, 32)
        self.ds = 2
        self.ncoils = 4
        self.dt = 5.
        self.n_readout_bins = 8

        np.random.seed(1)
        n = self.ds * self.data_shape[0]
        self.image_shape = (n, n, n)

        #a = np.pad(np.random.rand(n - 4, n - 4, n - 4), 2).astype(np.float64)
        #b = np.pad(np.random.rand(n - 4, n - 4, n - 4), 2).astype(np.float64)
        #self.x = np.stack([a, b], axis=-1)

        self.sens = np.random.rand(*(
            (self.ncoils, ) + self.data_shape)).astype(
                np.float64) + 1j * np.random.rand(*(
                    (self.ncoils, ) + self.data_shape)).astype(np.float64)

        self.readout_time = pynamr.TPIReadOutTime()
        self.kspace_part = pynamr.RadialKSpacePartitioner(
            self.data_shape, self.n_readout_bins)

    def test_monoexp_adjoint(self):

        gam = np.random.rand(*self.image_shape).astype(np.float64)

        m = pynamr.MonoExpDualTESodiumAcqModel(self.ds, self.sens, self.dt,
                                               self.readout_time,
                                               self.kspace_part)

        x = np.random.rand(*((self.image_shape) + (2, )))

        x_fwd = m.forward(x, gam)
        y = np.random.rand(*x_fwd.shape).astype(np.float64)
        y_back = m.adjoint(y, gam)

        self.assertTrue(np.isclose((x_fwd * y).sum(), (x * y_back).sum()))

    def test_biexp_adjoint(self):
        m = pynamr.TwoCompartmentBiExpDualTESodiumAcqModel(
            self.ds, self.sens, self.dt, self.readout_time, self.kspace_part,
            2, 20, 4, 16, 0.4, 0.2)

        x = np.random.rand(*((2, ) + (self.image_shape) + (2, )))

        x_fwd = m.forward(x)

        y = np.random.rand(*x_fwd.shape).astype(x_fwd.dtype)
        y_back = m.adjoint(y)

        self.assertTrue(np.isclose((x_fwd * y).sum(), (x * y_back).sum()))


if __name__ == '__main__':
    unittest.main()

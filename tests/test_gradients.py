import unittest
import numpy as np
try:
    import cupy as cp
except:
    cp = np

import pynamr


class TestGradients(unittest.TestCase):

    def setUp(self):
        self.data_shape = (32, 32, 32)
        self.ds = 2
        self.ncoils = 3
        self.dt = 5.
        self.noise_level = 0.1
        self.xp = np
        self.n_readout_bins = 8

        #----------------------------------------------------------------------
        np.random.seed(1)

        self.n_ds = self.data_shape[0]
        self.n = self.ds * self.n_ds
        self.image_shape = (self.n, self.n, self.n)

        self.sens = cp.random.rand(*(
            (self.ncoils, ) + self.data_shape)) + 1j * cp.random.rand(*(
                (self.ncoils, ) + self.data_shape))
        self.sens *= 1e-2

        self.gam = np.random.rand(*self.image_shape)

        readout_time = pynamr.TPIReadOutTime()
        kspace_part = pynamr.RadialKSpacePartitioner(self.data_shape,
                                                     self.n_readout_bins)

        # generate mono-exp. data
        self.mono_exp_model = pynamr.MonoExpDualTESodiumAcqModel(
            self.ds, self.sens, self.dt, readout_time, kspace_part, gam=self.gam)

        self.x = np.random.rand(*((1,) + (self.image_shape) + (2, )))
        self.y = self.mono_exp_model.forward(self.x)
        self.data = self.y + self.noise_level * np.abs(
            self.y).mean() * np.random.randn(*self.y.shape)

        # generate bi-exp dual comp data
        self.bi_exp_model = pynamr.TwoCompartmentBiExpDualTESodiumAcqModel(
            self.ds, self.sens, self.dt, readout_time, kspace_part, 2, 20, 4,
            16, 0.4, 0.2)

        self.x_bi = np.random.rand(*((2, ) + (self.image_shape) + (2, )))
        self.y_bi = self.bi_exp_model.forward(self.x_bi)
        self.data_bi = self.y_bi + self.noise_level * np.abs(
            self.y_bi).mean() * np.random.randn(*self.y_bi.shape)

    def test_data_fidelity_gradient_mono_exp(self, i=51, eps=1e-4, rtol=1e-3):
        # setup data fidelity loss
        loss = pynamr.DataFidelityLoss(self.mono_exp_model, self.data)

        # inital values
        x_0 = np.random.rand(*self.x.shape)
        gam_0 = np.random.rand(*self.gam.shape)

        # check gradients
        ll = loss(x_0, pynamr.CallingMode.XFIRST, gam_0)
        gx = loss.grad(x_0, pynamr.CallingMode.XFIRST, gam_0)
        gg = loss.grad(gam_0, pynamr.CallingMode.GAMFIRST, x_0)

        # test gradient with respect to Na image (real part)
        delta_x = np.zeros(x_0.shape)
        delta_x[0, i, i, i, 0] = eps
        self.assertTrue(
            np.isclose(
                gx[0, i, i, i, 0],
                (loss(x_0 + delta_x, pynamr.CallingMode.XFIRST, gam_0) - ll) /
                eps,
                rtol=rtol))

        # test gradient with respect to Na image (imag part)
        delta_x = np.zeros(x_0.shape)
        delta_x[0, i, i, i, 1] = eps
        self.assertTrue(
            np.isclose(
                gx[0, i, i, i, 1],
                (loss(x_0 + delta_x, pynamr.CallingMode.XFIRST, gam_0) - ll) /
                eps,
                rtol=rtol))

        # test gradient with respect to Gamma image
        delta_g = np.zeros(gam_0.shape)
        delta_g[i, i, i] = eps
        self.assertTrue(
            np.isclose(
                gg[i, i, i],
                (loss(x_0, pynamr.CallingMode.XFIRST, gam_0 + delta_g) - ll) /
                eps,
                rtol=rtol))

    def test_data_fidelity_gradient_bi_exp(self, i=51, eps=1e-4, rtol=1e-3):
        # setup data fidelity loss
        loss = pynamr.DataFidelityLoss(self.bi_exp_model, self.data_bi)

        # inital values
        x_0 = np.random.rand(*self.x_bi.shape)

        # check gradients
        ll = loss(x_0, pynamr.CallingMode.XFIRST)
        gx = loss.grad(x_0, pynamr.CallingMode.XFIRST)

        for ch in range(2):
          # test gradient with respect - real part
          delta_x = np.zeros(x_0.shape)
          delta_x[ch, i, i, i, 0] = eps
          self.assertTrue(
              np.isclose(
                  gx[ch, i, i, i, 0],
                  (loss(x_0 + delta_x, pynamr.CallingMode.XFIRST) - ll) /
                  eps,
                  rtol=rtol))

          # test gradient with respect - imag part
          delta_x = np.zeros(x_0.shape)
          delta_x[ch, i, i, i, 1] = eps
          self.assertTrue(
              np.isclose(
                  gx[ch, i, i, i, 1],
                  (loss(x_0 + delta_x, pynamr.CallingMode.XFIRST) - ll) /
                  eps,
                  rtol=rtol))


#    def test_total_gradient(self,
#                            i=51,
#                            eps=1e-4,
#                            rtol=1e-3,
#                            beta_x=1e-2,
#                            beta_gam=1e-2):
#        # setup data fidelity loss
#        data_fidelity_loss = pynamr.DataFidelityLoss(self.mono_exp_model,
#                                                     self.data)
#
#        # setup the Bowsher loss
#        nnearest = 2
#        aimg = np.random.rand(*self.image_shape)
#
#        s = np.array([[[0, 1, 0], [1, 1, 1], [0, 1, 0]],
#                      [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
#                      [[0, 1, 0], [1, 1, 1], [0, 1, 0]]])
#
#        nn_inds = np.zeros((np.prod(self.image_shape), nnearest),
#                           dtype=np.uint32)
#        pynamr.nearest_neighbors(aimg, s, nnearest, nn_inds)
#        nn_inds_adj = pynamr.is_nearest_neighbor_of(nn_inds)
#
#        bowsher_loss = pynamr.BowsherLoss(nn_inds, nn_inds_adj)
#
#        # setup the combined loss function
#        loss = pynamr.TotalLoss(data_fidelity_loss, bowsher_loss, bowsher_loss,
#                                beta_x, beta_gam)
#
#        # inital values
#        x_0 = np.random.rand(*self.x.shape)
#        gam_0 = np.random.rand(*self.gam.shape)
#
#        # check gradients
#        ll = loss(x_0, pynamr.CallingMode.XFIRST, gam_0)
#        gx = loss.grad(x_0, pynamr.CallingMode.XFIRST, gam_0)
#        gg = loss.grad(gam_0, pynamr.CallingMode.GAMFIRST, x_0)
#
#        # test gradient with respect to Na image (real part)
#        delta_x = np.zeros(x_0.shape)
#        delta_x[i, i, i, 0] = eps
#        self.assertTrue(
#            np.isclose(
#                gx[i, i, i, 0],
#                (loss(x_0 + delta_x, pynamr.CallingMode.XFIRST, gam_0) - ll) /
#                eps,
#                rtol=rtol))
#
#        # test gradient with respect to Na image (imag part)
#        delta_x = np.zeros(x_0.shape)
#        delta_x[i, i, i, 1] = eps
#        self.assertTrue(
#            np.isclose(
#                gx[i, i, i, 1],
#                (loss(x_0 + delta_x, pynamr.CallingMode.XFIRST, gam_0) - ll) /
#                eps,
#                rtol=rtol))
#
#        # test gradient with respect to Gamma image
#        delta_g = np.zeros(gam_0.shape)
#        delta_g[i, i, i] = eps
#        self.assertTrue(
#            np.isclose(
#                gg[i, i, i],
#                (loss(x_0, pynamr.CallingMode.XFIRST, gam_0 + delta_g) - ll) /
#                eps,
#                rtol=rtol))

    def test_bowsher_gradient(self, eps=1e-7, rtol=1e-3, atol=1e-4):
        nnearest = 2
        image_shape = (4, 5, 6)
        aimg = np.random.rand(*image_shape)
        timg = np.random.rand(*image_shape)

        s = np.array([[[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                      [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
                      [[0, 1, 0], [1, 1, 1], [0, 1, 0]]])

        nn_inds = np.zeros((np.prod(image_shape), nnearest), dtype=np.uint32)
        pynamr.nearest_neighbors(aimg, s, nnearest, nn_inds)
        nn_inds_adj = pynamr.is_nearest_neighbor_of(nn_inds)

        bl = pynamr.BowsherLoss(nn_inds, nn_inds_adj)

        c1 = bl(timg)

        g1 = bl.grad(timg)

        # check gradient numerically
        close = np.zeros(image_shape, dtype=np.uint8)

        for i in range(image_shape[0]):
            for j in range(image_shape[1]):
                for k in range(image_shape[2]):
                    delta = np.zeros(image_shape)
                    delta[i, j, k] = eps

                    g2 = (bl(timg + delta) - c1) / eps

                    close[i, j, k] = np.isclose(g1[i, j, k],
                                                g2,
                                                rtol=rtol,
                                                atol=atol)

        self.assertTrue(np.all(close))


#---------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

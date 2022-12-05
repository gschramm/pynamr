import unittest
import numpy as np
try:
    import cupy as cp
except:
    cp = np

from copy import deepcopy

import pynamr
from IPython.core.debugger import set_trace

class TestGradients(unittest.TestCase):

    def setUp(self):
        self.data_shape = (32, 32, 32)
        self.ds = 2
        self.ncoils = 3
        self.dt = 5.
        self.te1 = 0.5
        self.noise_level = 0.1
        self.xp = np
        self.n_readout_bins = 8
        self.n_ds = self.data_shape[0]
        self.n = self.ds * self.n_ds
        self.recon_shape = (self.n, self.n, self.n)
        self.spatial_n = 3

        # changing param
        self.ds_mode = 'kspace'
        if self.ds_mode == 'kspace':
            self.data_shape = self.recon_shape

        #----------------------------------------------------------------------
        np.random.seed(1)


        self.dim_info = {'data_shape': self.data_shape, 'recon_shape': self.recon_shape, 'ds': self.ds, 'ds_mode': self.ds_mode, 'spatial_n': self.spatial_n}

        self.sens = cp.random.rand(*(
            (self.ncoils, ) + self.recon_shape)) + 1j * cp.random.rand(*(
                (self.ncoils, ) + self.recon_shape))
        self.sens *= 1e-2

        self.gam = np.random.rand(*self.recon_shape)

        readout_time = pynamr.TPIReadOutTime()
        pad_factor = (self.ds if self.ds_mode=='kspace' else 1)
        kspace_part = pynamr.RadialKSpacePartitioner(self.data_shape, pad_factor, self.n_readout_bins)


        # generate mono-exp. data
        self.mono_exp_model = pynamr.MonoExpDualTESodiumAcqModel(self.dim_info,
                                                                 self.sens,
                                                                 self.dt,
                                                                 self.te1,
                                                                 readout_time,
                                                                 kspace_part)

        self.unknowns_mono = {pynamr.VarName.IMAGE: pynamr.Var(shape= self.recon_shape + (2,)),
        pynamr.VarName.GAMMA: pynamr.Var(shape=self.recon_shape, nb_comp=1, complex_var=False)}

        self.x = np.random.rand(*( (self.recon_shape) + (2, )))
        self.unknowns_mono[pynamr.VarName.IMAGE].value = self.x
        self.unknowns_mono[pynamr.VarName.GAMMA].value = self.gam

        self.y = self.mono_exp_model.forward(self.unknowns_mono)
        self.data = self.y + self.noise_level * np.abs(
            self.y).mean() * np.random.randn(*self.y.shape)

        # generate bi-exp dual comp data
        self.bi_exp_model = pynamr.TwoCompartmentBiExpDualTESodiumAcqModel(
            self.dim_info, self.sens, self.dt, self.te1, readout_time, kspace_part, 2, 20, 4,
            16, 0.4, 0.2)

        self.x_bi = np.random.rand(*((2, ) + (self.recon_shape) + (2, )))

        self.unknowns_bi = {pynamr.VarName.PARAM: pynamr.Var( shape= (2,) + self.recon_shape + (2,), nb_comp=2)}
        self.unknowns_bi[pynamr.VarName.PARAM].value = self.x_bi

        self.y_bi = self.bi_exp_model.forward(self.unknowns_bi)
        self.data_bi = self.y_bi + self.noise_level * np.abs(
            self.y_bi).mean() * np.random.randn(*self.y_bi.shape)


    def test_data_fidelity_gradient_mono_exp(self):

        # setup data fidelity loss
        loss = pynamr.DataFidelityLoss(self.mono_exp_model, self.data)

        # random values for variables
        self.unknowns_mono[pynamr.VarName.IMAGE].value = np.random.rand(*( (self.recon_shape) + (2, )))
        self.unknowns_mono[pynamr.VarName.GAMMA].value = np.random.rand(*(self.recon_shape))

        loss.gradient_test(self.unknowns_mono, pynamr.VarName.IMAGE) 
        loss.gradient_test(self.unknowns_mono, pynamr.VarName.GAMMA)


    def test_data_fidelity_gradient_bi_exp(self):

        # setup data fidelity loss
        loss = pynamr.DataFidelityLoss(self.bi_exp_model, self.data_bi)

        # random values for variables
        self.unknowns_bi[pynamr.VarName.PARAM].value =  np.random.rand(*((2, ) + (self.recon_shape) + (2, )))

        loss.gradient_test(self.unknowns_bi, pynamr.VarName.PARAM) 


    def test_total_gradient_mono_exp(self):

        # setup data fidelity loss
        data_fidelity_loss = pynamr.DataFidelityLoss(self.mono_exp_model,
                                                     self.data)

        # setup the Bowsher loss
        nnearest = 2
        aimg = np.random.rand(*self.recon_shape)

        s = np.array([[[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                      [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
                      [[0, 1, 0], [1, 1, 1], [0, 1, 0]]])

        nn_inds = np.zeros((np.prod(self.recon_shape), nnearest),
                           dtype=np.uint32)
        pynamr.nearest_neighbors(aimg, s, nnearest, nn_inds)
        nn_inds_adj = pynamr.is_nearest_neighbor_of(nn_inds)

        bowsher_loss = pynamr.BowsherLoss(nn_inds, nn_inds_adj)

        # setup the combined loss function
        beta_x = np.random.uniform(0.,100.)
        beta_gam = np.random.uniform(0.,100.)
        penalty_info = {pynamr.VarName.IMAGE: bowsher_loss, pynamr.VarName.GAMMA: bowsher_loss}
        beta_info = {pynamr.VarName.IMAGE: beta_x, pynamr.VarName.GAMMA: beta_gam}
        loss = pynamr.TotalLoss(data_fidelity_loss, penalty_info, beta_info)

        # random values for variables
        self.unknowns_mono[pynamr.VarName.IMAGE].value = np.random.rand(*( (self.recon_shape) + (2, )))
        self.unknowns_mono[pynamr.VarName.GAMMA].value = np.random.rand(*(self.recon_shape))

        loss.gradient_test(self.unknowns_mono, pynamr.VarName.IMAGE)
        loss.gradient_test(self.unknowns_mono, pynamr.VarName.GAMMA, rtol=1e-3)

    def test_total_gradient_bi_exp(self):

        # setup data fidelity loss
        data_fidelity_loss = pynamr.DataFidelityLoss(self.bi_exp_model,
                                                     self.data_bi)

        # setup the Bowsher loss
        nnearest = 2
        aimg = np.random.rand(*self.recon_shape)

        s = np.array([[[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                      [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
                      [[0, 1, 0], [1, 1, 1], [0, 1, 0]]])

        nn_inds = np.zeros((np.prod(self.recon_shape), nnearest),
                           dtype=np.uint32)
        pynamr.nearest_neighbors(aimg, s, nnearest, nn_inds)
        nn_inds_adj = pynamr.is_nearest_neighbor_of(nn_inds)

        bowsher_loss = pynamr.BowsherLoss(nn_inds, nn_inds_adj)

        # setup the combined loss function
        beta_x = np.random.uniform(0.,100.)
        penalty_info = {pynamr.VarName.PARAM: bowsher_loss}
        beta_info = {pynamr.VarName.PARAM: beta_x}
        loss = pynamr.TotalLoss(data_fidelity_loss, penalty_info, beta_info)

        # random values for variables
        self.unknowns_bi[pynamr.VarName.PARAM].value =  np.random.rand(*((2, ) + (self.recon_shape) + (2, )))

        loss.gradient_test(self.unknowns_bi, pynamr.VarName.PARAM)


    def test_bowsher_gradient(self):
        nnearest = 2
        recon_shape = (4, 5, 6)
        aimg = np.random.rand(*recon_shape)
        timg = np.random.rand(*recon_shape)

        s = np.array([[[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                      [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
                      [[0, 1, 0], [1, 1, 1], [0, 1, 0]]])

        nn_inds = np.zeros((np.prod(recon_shape), nnearest), dtype=np.uint32)
        pynamr.nearest_neighbors(aimg, s, nnearest, nn_inds)
        nn_inds_adj = pynamr.is_nearest_neighbor_of(nn_inds)

        bl = pynamr.BowsherLoss(nn_inds, nn_inds_adj)

        bl.gradient_test(timg)


#---------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

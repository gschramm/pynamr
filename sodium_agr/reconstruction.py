import numpy as np
import cupy as cp
import sigpy
import sigpy.mri


def channelwise_ifft_recon(data: np.ndarray,
                           k: np.ndarray,
                           grid_shape=(64, 64, 64),
                           field_of_view_cm=22.,
                           **kwargs) -> np.ndarray:

    # transfer k-space trajectory to GPU and convert to unitless
    k_d = cp.asarray(k.reshape(-1, 3)) * field_of_view_cm

    samp_dens = sigpy.gridding(cp.ones(k_d.shape[0], dtype=data.dtype), k_d,
                               grid_shape, **kwargs)
    kernel_gridded = sigpy.gridding(cp.ones(1, dtype=data.dtype),
                                    cp.zeros((1, 3), dtype=k_d.dtype),
                                    grid_shape, **kwargs)

    ifft_op = sigpy.linop.IFFT(grid_shape)
    kernel_ifft_d = ifft_op(kernel_gridded)

    x_ifft_d = cp.zeros((data.shape[0], ) + grid_shape, dtype=data.dtype)

    for i, d in enumerate(data):
        # transfer data to GPU
        d_d = cp.asarray(d.ravel())

        data_gridded = sigpy.gridding(d_d, k_d, grid_shape, **kwargs)

        data_gridded_corr = data_gridded.copy()
        data_gridded_corr[samp_dens > 0] /= samp_dens[samp_dens > 0]

        x_ifft_d[i, ...] = ifft_op(data_gridded_corr) / kernel_ifft_d

    return cp.asnumpy(x_ifft_d)


#----------------------------------------------------------------
#----------------------------------------------------------------
#----------------------------------------------------------------


def channelwise_lsq_recon(data: np.ndarray,
                          k: np.ndarray,
                          grid_shape=(64, 64, 64),
                          field_of_view_cm=22.,
                          max_iter=10,
                          **kwargs) -> np.ndarray:

    # transfer k-space trajectory to GPU and convert to unitless
    k_d = cp.asarray(k.reshape(-1, 3)) * field_of_view_cm
    A = sigpy.linop.NUFFT(grid_shape, k_d)

    x = cp.zeros((data.shape[0], ) + grid_shape, dtype=data.dtype)

    for i, d in enumerate(data):
        # transfer data to GPU
        d_d = cp.asarray(d.ravel())
        alg = sigpy.app.LinearLeastSquares(A, d_d, max_iter=max_iter, **kwargs)
        x[i, ...] = alg.run()

    return cp.asnumpy(x)


#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------


def regularized_sense_recon(data: np.ndarray,
                            coil_sens: np.ndarray,
                            k: np.ndarray,
                            field_of_view_cm=22.,
                            regulariztion: str = 'L2',
                            beta: float = 1e-1,
                            sigma: float = 1e-1,
                            operator_norm_squared: float | None = 1.,
                            max_iter: int = 100,
                            G: sigpy.linop.Linop | None = None,
                            **kwargs) -> np.ndarray:

    # send kspace trajectory and data to GPU
    k_d = cp.asarray(k.reshape(-1, 3)) * field_of_view_cm
    d_d = cp.asarray(data.reshape(data.shape[0], -1))

    # setup normalize Sense operator
    S = sigpy.mri.linop.Sense(cp.asarray(coil_sens), coord=k_d)

    # setup normalized gradient operator
    if G is None:
        G = (1 / np.sqrt(12)) * sigpy.linop.FiniteDifference(S.ishape)

    # setup prox for gradient norm
    if regulariztion == 'L2':
        proxg = sigpy.prox.L2Reg(G.oshape, lamda=beta)
        g = lambda z: float(beta * 0.5 * ((z * z.conj()).sum()).real)
    elif regulariztion == 'L1':
        proxg = sigpy.prox.L1Reg(G.oshape, lamda=beta)
        g = lambda z: float(beta * cp.abs(z).sum())
    else:
        raise ValueError('regularization must be L1 or L2')

    if operator_norm_squared is None:
        operator_norm_squared = sigpy.app.MaxEig(S.H * S,
                                                 dtype=data.dtype,
                                                 device=data.device).run()

    alg = sigpy.app.LinearLeastSquares(S,
                                       d_d,
                                       G=G,
                                       proxg=proxg,
                                       max_iter=max_iter,
                                       sigma=sigma,
                                       tau=0.99 * operator_norm_squared /
                                       sigma,
                                       g=g,
                                       **kwargs)
    x = cp.asnumpy(alg.run())

    return x

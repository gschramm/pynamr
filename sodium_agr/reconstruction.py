from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import cupy as cp
import sigpy
import sigpy.mri

from typing import Union
from scipy.optimize import fmin_l_bfgs_b

from operators import ApodizedNUFFT

# custom type indicating a numpy or cupy array
ndarray = Union[np.ndarray, cp.ndarray]


def channelwise_ifft_recon(data: np.ndarray,
                           k: np.ndarray,
                           grid_shape=(64, 64, 64),
                           field_of_view_cm=22.,
                           **kwargs) -> np.ndarray:
    """Channelwise gridding + IFFT recon of multi-coil data

    Parameters
    ----------
    data : np.ndarray
        array of (non-uniform) kspace data of shape (num_coils, num_time_points, num_readouts)
    k : np.ndarray
        array containing the kspace coordinates of the readouts in 1/cm
        shape (num_time_points, num_readouts, 3)
    grid_shape : tuple, optional
        shape of the image grid , by default (64, 64, 64)
    field_of_view_cm : _type_, optional
        the field of view in cm, by default 22.
    **kwargs:
        additional keyword arguments passed to sigpy.gridding

    Returns
    -------
    np.ndarray
        array of shape (num_coil, grid_shape)
    """

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


def channelwise_lsq_recon(data: np.ndarray,
                          k: np.ndarray,
                          grid_shape=(64, 64, 64),
                          field_of_view_cm=22.,
                          max_iter=10,
                          **kwargs) -> np.ndarray:
    """Channelwise iterative least squares recon of multi-coil data

    Parameters
    ----------
    data : np.ndarray
        array of (non-uniform) kspace data of shape (num_coils, num_time_points, num_readouts)
    k : np.ndarray
        array containing the kspace coordinates of the readouts in 1/cm
        shape (num_time_points, num_readouts, 3)
    grid_shape : tuple, optional
        shape of the image grid , by default (64, 64, 64)
    field_of_view_cm : _type_, optional
        the field of view in cm, by default 22.
    max_iter : int
        number of iterations, by default 10
    **kwargs:
        additional keyword arguments passed to sigpy.app.LinearLeastSquares

    Returns
    -------
    np.ndarray
        array of shape (num_coil, grid_shape)
    """

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


def regularized_sense_recon(data: np.ndarray,
                            coil_sens: np.ndarray,
                            k: np.ndarray,
                            field_of_view_cm: float = 22.,
                            regularization: str = 'L2',
                            beta: float = 1e-1,
                            sigma: float = 1e-1,
                            operator_norm_squared: float | None = 1.,
                            max_iter: int = 100,
                            G: sigpy.linop.Linop | None = None,
                            u: np.ndarray | None = None,
                            **kwargs) -> tuple[np.ndarray, np.ndarray | None]:
    """regularized iterative Sense recon of multicoil data

    Parameters
    ----------
    data : np.ndarray
        array of (non-uniform) kspace data of shape (num_coils, num_time_points, num_readouts)
    coil_sens : np.ndarray
        array of coil sensitivity maps of shape (num_coils, nx, ny, nz)
    k : np.ndarray
        array containing the kspace coordinates of the readouts in 1/cm
        shape (num_time_points, num_readouts, 3)
    field_of_view_cm : float, optional
        the field of view in cm, by default 22.
    regularization : str, optional
        norm used for regularization ('L1' or 'L2'), by default 'L2'
    beta : float, optional
        regularization weight, by default 1e-1
    sigma : float, optional
        sigma step size for PDHG, by default 1e-1
    operator_norm_squared : float | None, optional
        squared norm of the operator used to calcucate 
        tau step size in PDHG, by default 1.
    max_iter : int, optional
        maximum number of iterations, by default 100
    G : sigpy.linop.Linop | None, optional
        operator used for regularization, by default None
    u : np.ndarray | None, optional
        initial value of the dual variable in PDHG, by default None

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        single array containing the reconstructed image (if G is None)
        or tuple of the reconstructed image and the dual variable u
        (if G is not None)
    """
    # send kspace trajectory and data to GPU
    k_d = cp.asarray(k.reshape(-1, 3)) * field_of_view_cm
    d_d = cp.asarray(data.reshape(data.shape[0], -1))

    # setup normalize Sense operator
    S = sigpy.mri.linop.Sense(cp.asarray(coil_sens), coord=k_d)

    # setup normalized gradient operator
    if G is None:
        G = (1 / np.sqrt(12)) * sigpy.linop.FiniteDifference(S.ishape)

    # setup prox for gradient norm
    if regularization == 'L2':
        proxg = sigpy.prox.L2Reg(G.oshape, lamda=beta)
        g = lambda z: float(beta * 0.5 * ((z * z.conj()).sum()).real)
    elif regularization == 'L1':
        proxg = sigpy.prox.L1Reg(G.oshape, lamda=beta)
        g = lambda z: float(beta * cp.abs(z).sum())
    else:
        raise ValueError('regularization must be L1 or L2')

    if operator_norm_squared is None:
        operator_norm_squared = sigpy.app.MaxEig(S.H * S,
                                                 dtype=data.dtype,
                                                 device=data.device).run()

    app = sigpy.app.LinearLeastSquares(S,
                                       d_d,
                                       G=G,
                                       proxg=proxg,
                                       max_iter=max_iter,
                                       sigma=sigma,
                                       tau=0.99 * operator_norm_squared /
                                       sigma,
                                       g=g,
                                       **kwargs)
    if G is not None:
        if u is not None:
            app.alg.u = cp.asarray(u)

    x = cp.asnumpy(app.run())

    if G is not None:
        return x, cp.asnumpy(app.alg.u)
    else:
        return x, None


#---------------------------------------------------------------


def data_fidelity_gradient_r(x: ndarray, A_1: ApodizedNUFFT,
                             A_2: ApodizedNUFFT, d_1: ndarray,
                             d_2: ndarray) -> ndarray:
    """gradient with respect to the ratio image of the dual data fidelity

    Parameters
    ----------
    x : ndarray
        containing the sodium image
    A_1, A_2 : ApodizedNUFFT
        apodized NUFFT operator for 1st / 2nd echo
    d_1, d_2 : cp.ndarray
        non-uniform data of 1st / 2nd echo

    Returns
    -------
    ndarray
        containg the gradient with respect to r
    """
    xp = sigpy.backend.get_array_module(x)
    g_r = xp.zeros(A_1.ishape, dtype=A_1.r.dtype)

    for i in range(A_1.num_time_bins):
        inds = A_1.get_split_inds(i)

        A_1i = A_1.get_Ai(i)
        A_2i = A_2.get_Ai(i)

        g_r += A_1.tau[i] * (A_1.r**(A_1.tau[i] - 1)) * (
            x.conj() * A_1i.H(A_1i(x) - d_1[:, inds, :])).real
        g_r += A_2.tau[i] * (A_2.r**(A_2.tau[i] - 1)) * (
            x.conj() * A_2i.H(A_2i(x) - d_2[:, inds, :])).real

    return g_r


#---------------------------------------------------------------
def r_cost_wrapper(r_flat: np.ndarray, x: cp.ndarray, A_1: ApodizedNUFFT,
                   A_2: ApodizedNUFFT, d_1: cp.ndarray, d_2: cp.ndarray,
                   G: sigpy.linop.Linop, beta: float) -> float:
    """wrapper around the data fidelity cost function that can be used
       by scipy.optimize.fmin_l_bfgs_b

    Parameters
    ----------
    r_flat : np.ndarray
        flattened array containing the ratio image
    x : cp.ndarray
        array containing the sodium image
    A_1, A_2 : ApodizedNUFFT
        apodized NUFFT operator for 1st / 2nd echo
    d_1, d_2 : cp.ndarray
        non-uniform data of 1st / 2nd echo
    G : sigpy.linop.Linop
        linear operator used for regularization
    beta : float
        regularization weight

    Returns
    -------
    float
        the data fidelity
    """
    r_init = A_1.r.copy()
    r = cp.asarray(r_flat.reshape(A_1.ishape))
    A_1.r = r
    A_2.r = r

    data_fid = 0.5 * float((cp.abs(A_1(x) - d_1)**2).sum() +
                           (cp.abs(A_2(x) - d_2)**2).sum())
    prior = 0.5 * beta * float((G(r)**2).sum())

    A_1.r = r_init
    A_2.r = r_init

    return (data_fid + prior)


def r_gradient_wrapper(r_flat: np.ndarray, x: cp.ndarray, A_1: ApodizedNUFFT,
                       A_2: ApodizedNUFFT, d_1: cp.ndarray, d_2: cp.ndarray,
                       G: sigpy.linop.Linop, beta: float) -> np.ndarray:
    """wrapper around the r gradient data fidelity cost function that can be used
       by scipy.optimize.fmin_l_bfgs_b

    Parameters
    ----------
    r_flat : np.ndarray
        flattened array containing the ratio image
    x : cp.ndarray
        array containing the sodium image
    A_1, A_2 : ApodizedNUFFT
        apodized NUFFT operator for 1st / 2nd echo
    d_1, d_2 : cp.ndarray
        non-uniform data of 1st / 2nd echo
    G : sigpy.linop.Linop
        linear operator used for regularization
    beta : float
        regularization weight

    Returns
    -------
    np.ndarray
        the (flattned) gradient of the data fidelity with respect to r
    """

    r_init = A_1.r.copy()
    r = cp.asarray(r_flat.reshape(A_1.ishape))
    A_1.r = r
    A_2.r = r

    data_grad = data_fidelity_gradient_r(x, A_1, A_2, d_1, d_2)
    prior_grad = beta * G.H(G(r))

    A_1.r = r_init
    A_2.r = r_init

    return cp.asnumpy((data_grad + prior_grad).ravel())


def dual_echo_sense_with_decay_estimation(
        data_1: np.ndarray,
        data_2: np.ndarray,
        sampling_time_us: float,
        TE1_ms: float,
        TE2_ms: float,
        coil_sens_1: np.ndarray,
        coil_sens_2: np.ndarray,
        k: np.ndarray,
        x0: np.ndarray,
        u0: np.ndarray,
        r0: np.ndarray,
        G: sigpy.linop.Linop,
        field_of_view_cm=22.,
        regularization: str = 'L1',
        beta: float = 3e-3,
        sigma: float = 1e-1,
        max_iter: int = 100,
        max_outer_iter: int = 20,
        num_time_bins: int = 64,
        beta_r: float = 1e-3,
        save_intermed: bool = True,
        **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Dual echo AGR with T2* decay estimation and modeling

    Parameters
    ----------
    data_1, data_2 : np.ndarray
        non-uniform (TPI) complex kspace data
        of shape (num_coils, num_time_points, num_readouts)
    sampling_time_us : float
        acquisition sampling time in microseconds
    TE1_ms, TE2_ms : float
        1st / 2nd echo time in milli seconds
    coil_sens_1, coil_sens_2 : np.ndarray
        coil sensitivity maps of shape (num_coils, nx, ny, nz)
        for 1st / 2nd acquisition
        note that these should include a phase factor that accounts
        for the phase differences in the 1st / 2nd echo images
    k : np.ndarray
        array of kspace kooridnates of shape (num_time_points, num_readouts, 3)
    x0 : np.ndarray
        initial value for the image to be reconstructed
    u0 : np.ndarray
        initial value for the dual variable of the PDHG
    r0 : np.ndarray
        initial value for the ratio image between 2nd and 1st echo
    G : sigpy.linop.Linop
        "gradient" operator for the regularization
        can be projector gradient to get AGRs
    field_of_view_cm : _type_, optional
        image field of view in cm, by default 22.
    regularization : str, optional
        norm used for image regularization ('L1' or 'L2'), by default 'L1'
    beta : float, optional
        weight for image prior, by default 3e-3
    sigma : float, optional
        sigma stepsize of PDHG, by default 1e-1
    max_iter : int, optional
        number of PDHG inner iterations to update the image, by default 100
    max_outer_iter : int, optional
        number of outer iterations, by default 20
    num_time_bins : int, optional
        number of time bins to model decay, by default 64
    beta_r : float, optional
        weight of L2 gradient prior for the ration image, by default 1e-3
    save_intermed : bool, optional
        save intermediate results in a temporary directory, by default True

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        the reconstructed image and the ratio image
    """

    d_1 = cp.asarray(data_1)
    d_2 = cp.asarray(data_2)

    x = cp.asarray(x0)
    u = cp.asarray(u0)
    r = cp.asarray(r0)

    Flist_1 = [
        sigpy.mri.linop.Sense(cp.asarray(coil_sens_1), cp.asarray(kchunk)) for
        kchunk in np.array_split(k * field_of_view_cm, num_time_bins, axis=0)
    ]

    Flist_2 = [
        sigpy.mri.linop.Sense(cp.asarray(coil_sens_2), cp.asarray(kchunk)) for
        kchunk in np.array_split(k * field_of_view_cm, num_time_bins, axis=0)
    ]

    t_read_1_ms = np.arange(k.shape[0]) * sampling_time_us * (1e-3) + TE1_ms
    t_read_2_ms = np.arange(k.shape[0]) * sampling_time_us * (1e-3) + TE2_ms

    tau_1 = [
        x.mean()
        for x in np.array_split(t_read_1_ms / (TE2_ms - TE1_ms), num_time_bins)
    ]
    tau_2 = [
        x.mean()
        for x in np.array_split(t_read_2_ms / (TE2_ms - TE1_ms), num_time_bins)
    ]

    A_1 = ApodizedNUFFT(Flist_1, r, tau_1)
    A_2 = ApodizedNUFFT(Flist_2, r, tau_2)

    # setup prox for gradient norm
    if regularization == 'L2':
        proxg = sigpy.prox.L2Reg(G.oshape, lamda=beta)
    elif regularization == 'L1':
        proxg = sigpy.prox.L1Reg(G.oshape, lamda=beta)
    else:
        raise ValueError('regularization must be L1 or L2')

    if save_intermed:
        tempdir = tempfile.TemporaryDirectory()
        temppath = Path(tempdir.name)

    r_bounds = r.size * [(0.01, 1)]

    for i_outer in range(max_outer_iter):
        print(
            f'dual echo AGR with decay est. outer iteration {(i_outer+1):03} / {max_outer_iter:03}'
        )

        # use L-BFGS_B to update the ratio image
        rf = cp.asnumpy(A_1.r.copy()).flatten()
        res = fmin_l_bfgs_b(r_cost_wrapper,
                            rf,
                            fprime=r_gradient_wrapper,
                            args=(x, A_1, A_2, d_1, d_2, G, beta_r),
                            maxiter=10,
                            bounds=r_bounds,
                            disp=1)

        r_new = cp.asarray(res[0].reshape(A_1.ishape))

        A_1.r = r_new
        A_2.r = r_new

        #-----------------------------------------------------------------------
        app = sigpy.app.LinearLeastSquares(sigpy.linop.Vstack([A_1, A_2],
                                                              axis=0),
                                           cp.concatenate((d_1, d_2), 0),
                                           G=G,
                                           proxg=proxg,
                                           x=x,
                                           max_iter=max_iter,
                                           sigma=sigma,
                                           tau=0.99 * sigma / 1.5,
                                           **kwargs)

        app.alg.u = u
        x = app.run()

        # save the dual variable for the initialization of the next PDHG run
        u = app.alg.u.copy()

        if save_intermed:
            print(f'saving intermediate results to {temppath}')
            cp.save(temppath / f'r_{i_outer+1}.npy', A_1.r)
            cp.save(temppath / f'x_{i_outer+1}.npy', x)

    return cp.asnumpy(x), cp.asnumpy(r_new)
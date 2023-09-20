from __future__ import annotations

import sigpy
import math
import numpy as np
import cupy as cp

from typing import Union
from utils import kb_rolloff

class NUFFTT2starDualEchoModel:

    def __init__(self,
                 ishape: tuple[int, int, int],
                 k_1_cm: np.ndarray,
                 field_of_view_cm: float = 22.,
                 acq_sampling_time_ms: float = 0.016,
                 time_bin_width_ms: float = 0.25,
                 scale: float = 1.,
                 echo_time_1_ms: float = 0.5,
                 echo_time_2_ms: float = 5,
                 nufft_kwargs=None) -> None:
        """sigpy (dual echo) forward nufft operator including monoexp. T2* decay modeling

        Parameters
        ----------
        ishape : tuple[int, int, int]
            shape of the input image
        k_1_cm : np.ndarray
            input kx, ky, kz coordinates - shape: (num_samples, num_readouts, 3)
            units 1/cm
        field_of_view_cm : float, optional
            field of view in cm, by default 220.
        acq_sampling_time_ms : float, optional
            samplignt time during acquisition in ms, by default 0.01
        time_bin_width_ms : float, optional
            time bin width for modeling T2* decay, by default 0.25
        scale : float, optional
            scale of the forward operator, by default 0.03
        echo_time_1_ms : float, optional
            first echo time in ms, by default 0.5
        echo_time_2_ms : float, optional
            second echo time in ms, by default 5.
        """
        self._ishape = ishape
        self._scale = scale
        if nufft_kwargs is None:
            self._nufft_kwargs = {}
        else:
            self._nufft_kwargs = nufft_kwargs

        self._acq_sampling_time_ms = acq_sampling_time_ms
        self._time_bin_width_ms = time_bin_width_ms
        self._echo_time_1_ms = echo_time_1_ms
        self._echo_time_2_ms = echo_time_2_ms

        self._time_bins_inds = np.array_split(
            np.arange(k_1_cm.shape[0]),
            math.ceil(k_1_cm.shape[0] /
                      (self._time_bin_width_ms / self._acq_sampling_time_ms)))

        self._coords = []

        for _, time_bin_inds in enumerate(self._time_bins_inds):
            chunk_coords_1_cm = k_1_cm[time_bin_inds, :, :].reshape(
                -1, k_1_cm.shape[-1])

            self._coords.append(chunk_coords_1_cm * field_of_view_cm)

        self._x = None
        self._dual_echo_data = None

        self._phase_factor_1 = None
        self._phase_factor_2 = None

        self._data_weights_1 = None
        self._data_weights_2 = None

    @property
    def x(self) -> Union[None, cp.ndarray]:
        return self._x

    @x.setter
    def x(self, value: Union[None, cp.ndarray]) -> None:
        self._x = value

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value

    @property
    def phase_factor_1(self) -> Union[None, cp.ndarray]:
        return self._phase_factor_1

    @phase_factor_1.setter
    def phase_factor_1(self, value: Union[None, cp.ndarray]) -> None:
        self._phase_factor_1 = value

    @property
    def phase_factor_2(self) -> Union[None, cp.ndarray]:
        return self._phase_factor_2

    @phase_factor_2.setter
    def phase_factor_2(self, value: Union[None, cp.ndarray]) -> None:
        self._phase_factor_2 = value

    @property
    def data_weights_1(self) -> Union[None, cp.ndarray]:
        return self._data_weights_1

    @data_weights_1.setter
    def data_weights_1(self, value: Union[None, cp.ndarray]) -> None:
        self._data_weights_1 = value

    @property
    def data_weights_2(self) -> Union[None, cp.ndarray]:
        return self._data_weights_2

    @data_weights_2.setter
    def data_weights_2(self, value: Union[None, cp.ndarray]) -> None:
        self._data_weights_2 = value

    @property
    def dual_echo_data(self) -> Union[None, cp.ndarray]:
        return self._dual_echo_data

    @dual_echo_data.setter
    def dual_echo_data(self, value: Union[None, cp.ndarray]) -> None:
        self._dual_echo_data = value

    def get_operators_wo_decay_model(
            self) -> tuple[sigpy.linop.Linop, sigpy.linop.Linop]:
        op1 = self._scale * sigpy.linop.NUFFT(
            self._ishape, cp.vstack(self._coords), **self._nufft_kwargs)

        op2 = self._scale * sigpy.linop.NUFFT(
            self._ishape, cp.vstack(self._coords), **self._nufft_kwargs)

        if self._phase_factor_1 is not None:
            op1 = op1 * sigpy.linop.Multiply(self._ishape,
                                             self._phase_factor_1)

        if self._phase_factor_2 is not None:
            op2 = op2 * sigpy.linop.Multiply(self._ishape,
                                             self._phase_factor_2)

        if self._data_weights_1 is not None:
            op1 = sigpy.linop.Multiply(op1.oshape, self._data_weights_1) * op1

        if self._data_weights_2 is not None:
            op2 = sigpy.linop.Multiply(op2.oshape, self._data_weights_2) * op2

        return op1, op2

    def get_operators_w_decay_model(
            self,
            r: cp.ndarray) -> tuple[sigpy.linop.Linop, sigpy.linop.Linop]:
        """NUFFT operators for dual echo including mono-exponential decay model

        Parameters
        ----------
        r : cp.ndarray
            ratio image (2nd / 1st echo image)

        Returns
        -------
        Union[sigpy.linop.Linop, sigpy.linop.Linop]
        """
        op1s = []
        op2s = []
        for i, time_bin_inds in enumerate(self._time_bins_inds):
            #setup the decay image
            readout_time_1_ms = self._echo_time_1_ms + time_bin_inds.mean(
            ) * self._acq_sampling_time_ms
            readout_time_2_ms = self._echo_time_2_ms + time_bin_inds.mean(
            ) * self._acq_sampling_time_ms

            n_1 = ((readout_time_1_ms) /
                   (self._echo_time_2_ms - self._echo_time_1_ms))
            n_2 = ((readout_time_2_ms) /
                   (self._echo_time_2_ms - self._echo_time_1_ms))

            op1s.append(
                sigpy.linop.NUFFT(self._ishape, self._coords[i], **
                                  self._nufft_kwargs) *
                sigpy.linop.Multiply(self._ishape, r**n_1))

            op2s.append(
                sigpy.linop.NUFFT(self._ishape, self._coords[i], **
                                  self._nufft_kwargs) *
                sigpy.linop.Multiply(self._ishape, r**n_2))

        operator1 = self._scale * sigpy.linop.Vstack(op1s)
        operator2 = self._scale * sigpy.linop.Vstack(op2s)

        if self._phase_factor_1 is not None:
            operator1 = operator1 * sigpy.linop.Multiply(
                self._ishape, self._phase_factor_1)

        if self._phase_factor_2 is not None:
            operator2 = operator2 * sigpy.linop.Multiply(
                self._ishape, self._phase_factor_2)

        if self._data_weights_1 is not None:
            operator1 = sigpy.linop.Multiply(operator1.oshape,
                                             self._data_weights_1) * operator1

        if self._data_weights_2 is not None:
            operator2 = sigpy.linop.Multiply(operator2.oshape,
                                             self._data_weights_2) * operator2

        return operator1, operator2

    def data_fidelity_gradient_r(self, r: cp.ndarray) -> cp.ndarray:
        """calculate the gradient of the dual echo data fidelity w.r.t to the ratio image

           The gradient is given by n * r*(n-1) Re (x.conj() A^H d)
           where d is the difference between current expectation and the model and
           A^H is the adjoint of the dual echo NUFFT operator without decay model
        
           Before calling the method, you need to set the x (current image estimate) 
           and dual_echo_data properties 

        Parameters
        ----------
        r : cp.ndarray
            the current ratio image

        Returns
        -------
        cp.ndarray
        """

        if self._x is None:
            raise ValueError("x is not set")

        if self._dual_echo_data is None:
            raise ValueError("dual echo data is not set")

        # calculate the difference between the current expectation and the model
        A_e1, A_e2 = self.get_operators_w_decay_model(r)
        A = sigpy.linop.Vstack([A_e1, A_e2])
        diff = A(self._x) - self._dual_echo_data

        # account for data weights
        if self._data_weights_1 is not None:
            diff[:A_e1.oshape[0]] *= self._data_weights_1
        if self._data_weights_2 is not None:
            diff[A_e1.oshape[0]:] *= self._data_weights_2

        del A
        del A_e1
        del A_e2

        # setup the operator we need to calculate the gradient
        f1s = []
        f2s = []
        for i, time_bin_inds in enumerate(self._time_bins_inds):
            #setup the decay image
            readout_time_1_ms = self._echo_time_1_ms + time_bin_inds.mean(
            ) * self._acq_sampling_time_ms
            readout_time_2_ms = self._echo_time_2_ms + time_bin_inds.mean(
            ) * self._acq_sampling_time_ms

            n_1 = ((readout_time_1_ms) /
                   (self._echo_time_2_ms - self._echo_time_1_ms))
            n_2 = ((readout_time_2_ms) /
                   (self._echo_time_2_ms - self._echo_time_1_ms))

            f1s.append(
                sigpy.linop.Multiply(self._ishape,
                                     n_1 * (r**(n_1 - 1)) * self._x.conj()) *
                sigpy.linop.NUFFT(self._ishape, self._coords[i], **
                                  self._nufft_kwargs).H)

            f2s.append(
                sigpy.linop.Multiply(self._ishape,
                                     n_2 * (r**(n_2 - 1)) * self._x.conj()) *
                sigpy.linop.NUFFT(self._ishape, self._coords[i], **
                                  self._nufft_kwargs).H)

        f1s = self._scale * sigpy.linop.Hstack(f1s)
        f2s = self._scale * sigpy.linop.Hstack(f2s)

        if self._phase_factor_1 is not None:
            f1s = sigpy.linop.Multiply(self._ishape,
                                       self._phase_factor_1.conj()) * f1s

        if self._phase_factor_2 is not None:
            f2s = sigpy.linop.Multiply(self._ishape,
                                       self._phase_factor_2.conj()) * f2s

        h_op = sigpy.linop.Hstack([f1s, f2s])

        return cp.real(h_op(diff))


def projected_gradient_operator(ishape: tuple[int, ...],
                                prior_image: cp.ndarray,
                                eta: float = 0.) -> sigpy.linop.Linop:
    """Projected gradient operator as defined in https://doi.org/10.1137/15M1047325.
       Gradient operator that return the component of a gradient that is orthogonal 
       to a joint gradient field (derived from a prior image)

    Parameters
    ----------
    ishape : tuple[int, ...]
        input image shape
    prior_image : Union[np.ndarray, cp.ndarray]
        the prior image used to calcuate the joint gradient field for the projection

    Returns
    -------
    sigpy.linop.Linop
    """

    # normalized "normal" gradient operator
    G = (1 / np.sqrt(4 * len(ishape))) * sigpy.linop.FiniteDifference(
        ishape, axes=None)

    xi = G(prior_image)

    # normalize the real and imaginary part of the joint gradient field
    real_norm = cp.sqrt(cp.linalg.norm(xi.real, axis=0)**2 + eta**2)
    imag_norm = cp.sqrt(cp.linalg.norm(xi.imag, axis=0)**2 + eta**2)

    ir = cp.where(real_norm > 0)
    ii = cp.where(imag_norm > 0)

    for i in range(xi.shape[0]):
        xi[i, ...].real[ir] /= real_norm[ir]
        xi[i, ...].imag[ii] /= imag_norm[ii]

    M = sigpy.linop.Multiply(G.oshape, xi)
    S = sigpy.linop.Sum(M.oshape, (0, ))
    I = sigpy.linop.Identity(M.oshape)

    # projection operator
    P = I - (M.H * S.H * S * M)

    # projected gradient operator
    PG = P * G

    return PG

class NUFFT_TPI_BiexpModel:

    def __init__(self,
                 ishape: tuple[int, int, int],
                 k_1_cm: np.ndarray,
                 T2star_short_map: np.ndarray,
                 T2star_long_map: np.ndarray,
                 field_of_view_cm: float = 22.,
                 acq_sampling_time_ms: float = 0.01,
                 time_bin_width_ms: float = 0.25,
                 echo_time_ms: float = 0.5) -> None:
        """sigpy simple forward TPI NUFFT operator including bi-exp. T2* decay modeling

        Parameters
        ----------
        ishape : tuple[int, int, int]
            shape of the input image
        k_1_cm : np.ndarray
            input kx, ky, kz coordinates - shape: (num_samples, num_readouts, 3)
            units 1/cm
        T2star_short_map: np.ndarray
            spatial map of short T2* component
        T2star_long_map: np.ndarray
            spatial map of long T2* component
        field_of_view_cm : float, optional
            field of view in cm, by default 220.
        acq_sampling_time_ms : float, optional
            samplignt time during acquisition in ms, by default 0.01
        time_bin_width_ms : float, optional
            time bin width for modeling T2* decay, by default 0.25
        echo_time_1_ms : float, optional
            first echo time in ms, by default 0.5
        """
        self._ishape = ishape

        self._acq_sampling_time_ms = acq_sampling_time_ms
        self._time_bin_width_ms = time_bin_width_ms
        self._echo_time_ms = echo_time_ms

        self._time_bins_inds = np.array_split(
            np.arange(k_1_cm.shape[0]),
            math.ceil(k_1_cm.shape[0] /
                      (self._time_bin_width_ms / self._acq_sampling_time_ms)))

        self._coords = []

        # sort coordinates into time bins
        for _, time_bin_inds in enumerate(self._time_bins_inds):
            chunk_coords_1_cm = k_1_cm[time_bin_inds, :, :].reshape(
                -1, k_1_cm.shape[-1])

            self._coords.append(chunk_coords_1_cm * field_of_view_cm)

        # send T2* maps to GPU
        self.T2star_short_map = cp.asarray(T2star_short_map)
        self.T2star_long_map = cp.asarray(T2star_long_map)

    def get_operator_wo_decay_model(self) -> sigpy.linop.Linop:
        op = sigpy.linop.NUFFT(self._ishape, cp.vstack(self._coords))
        return op

    def get_operator_w_decay_model(self) -> sigpy.linop.Linop:
        """NUFFT operators for TPI including bi-exponential decay model

        Returns
        -------
        sigpy.linop.Linop
        """
        ops = []
        for i, time_bin_inds in enumerate(self._time_bins_inds):
            #setup the decay image
            readout_time_ms = self._echo_time_ms + time_bin_inds.mean(
            ) * self._acq_sampling_time_ms

            # fixed ratio between short and long component
            decay = 0.6 * cp.exp(-readout_time_ms / self.T2star_short_map)
            decay +=  0.4 * cp.exp(-readout_time_ms / self.T2star_long_map)
            ops.append(
                sigpy.linop.NUFFT(self._ishape, self._coords[i]) *
                sigpy.linop.Multiply(self._ishape, decay))

        op = sigpy.linop.Vstack(ops)

        return op

def recon_empirical_grid_and_ifft(k_data: cp.ndarray, nufft_k: cp.ndarray, grid_shape: tuple) -> cp.ndarray:
    """
        Empirical gridding + IFFT reconstruction:
        sigpy gridding function, Kaiser-Bessel kernel with optimal parameters (Jackson et al.)
        No explicit sampling density correction
        Empirical normalization including sampling density and interpolation coefficients

        Parameters
        ----------
        k_data : 1d cupy array
            input flattened nufft k-space data
        nufft_k : cupy array of shape (nb_samples, 3)
            corresponding flattened k-space coordinates as used in sigpy nufft
        grid_shape : tuple
            the grid shape

        Returns
        ----------
        recon : cupy array of grid_shape shape, unscaled reconstructed image

    """

    # interpolation kernel parameters
    kernel = 'kaiser_bessel'
    width = 2
    param = 9.14

    # grid data
    data_gridded = sigpy.gridding(k_data,
                                 nufft_k,
                                 grid_shape,
                                 kernel=kernel,
                                 width=width,
                                 param=param)

    # normalization coefficients taking empirically into account
    # sampling density and interpolation coefficients
    norm_coefs = sigpy.gridding(cp.ones_like(k_data),
                           nufft_k,
                           grid_shape,
                           kernel=kernel,
                           width=width,
                           param=param)
    data_gridded_corr = data_gridded
    data_gridded_corr[norm_coefs > 0] /= norm_coefs[norm_coefs > 0]

    # sigpy ifft operator, though just cupy ifft with ortho norm
    ifft_op = sigpy.linop.IFFT(grid_shape, center=False)

    # phase correction field to account phase definition in numpy's fft
    tmp_x = np.arange(grid_shape[0])
    TMP_X, TMP_Y, TMP_Z = np.meshgrid(tmp_x, tmp_x, tmp_x)
    phase_corr = cp.asarray(((-1)**TMP_X) * ((-1)**TMP_Y) * ((-1)**TMP_Z))
    data_gridded_corr *= phase_corr

    # apply ifft
    recon = ifft_op(data_gridded_corr)

    # correction in image space for the KB kernel applied in k-space
    tmp_x = cp.linspace(-width / 2, width / 2, grid_shape[0])
    TMP_X, TMP_Y, TMP_Z = cp.meshgrid(tmp_x, tmp_x, tmp_x)
    R = cp.sqrt(TMP_X**2 + TMP_Y**2 + TMP_Z**2)
    R = cp.clip(R, 0, tmp_x.max())
    #TODO: understand why factor 1.6 is needed when regridding in 3D
    interpolation_correction_field = kb_rolloff(1.6*R, param)
    interpolation_correction_field /= interpolation_correction_field.max()
    recon = recon / interpolation_correction_field

    return recon

     

def recon_gridding(k_data: cp.ndarray,
                   nufft_k_coords: cp.ndarray,
                   grid_shape: tuple,
                   sample_density_corr: cp.ndarray) -> cp.ndarray:
    """
        Recon using standard gridding with known sampling density:
        Sampling density compensation + gridding (sigpy nufft adjoint) + IFFT

        Parameters
        ----------
        k_data : 1d cupy array, input flattened nufft k-space data
        nufft_k_coords : cupy array of shape (nb_samples, 3)
                            corresponding flattened k-space coordinates as used in sigpy nufft
        grid_shape : tuple
                        the Cartesian grid shape
        sample_density_corr : cupy array of k-space data shape
                                 multiplicative sampling density compensation

        Returns
        ----------
        recon : cupy array of grid_shape shape, reconstructed image

    """

    # sampling density compensation
    k_data_corrected = k_data * sample_density_corr

    # nufft adjoint
    recon = sigpy.nufft_adjoint(k_data_corrected,
                                 nufft_k_coords,
                                 grid_shape)

    return recon


def recon_tpi_iterative_nufft(k_data: cp.ndarray,
                          recon_shape,
                          k_1_cm,
                          field_of_view_cm = 22.,
                          acq_sampling_time_ms = 0.01,
                          time_bin_width_ms = 0.25,
                          echo_time_ms = 0.5,
                          beta_reg = 0.) -> cp.ndarray:
    """
        Simplest possible iterative recon of raw TPI data:
        default sigpy MLS conjugate gradient + Tikhonov regularization

        Parameters
        ----------
        k_data : cupy array,
            raw TPI data
        recon_shape : tuple[int, int, int]
            shape of the input image
        k_1_cm : np.ndarray
            input kx, ky, kz coordinates - shape: (num_samples, num_readouts, 3)
            units 1/cm
        field_of_view_cm : float, optional
            field of view in cm, by default 220.
        acq_sampling_time_ms : float, optional
            samplignt time during acquisition in ms, by default 0.01
        time_bin_width_ms : float, optional
            time bin width for modeling T2* decay, by default 0.25
        echo_time_ms : float, optional
            echo time in ms, by default 0.5
        beta_reg : float, optional
            Tikhonov regularization parameter

        Returns
        ----------
        recon : cupy array of grid_shape shape
            reconstructed image

    """

    # we don't know the decay
    nodecay_T2 = 1e7 * np.ones(recon_shape, np.float64)
    acq_model = NUFFT_TPI_BiexpModel(recon_shape,
                                     k_1_cm,
                                     nodecay_T2,
                                     nodecay_T2,
                                     field_of_view_cm=field_of_view_cm,
                                     acq_sampling_time_ms=acq_sampling_time_ms,
                                     time_bin_width_ms=time_bin_width_ms,
                                     echo_time_ms=echo_time_ms)

    # forward operator based on TPI trajectory
    A = acq_model.get_operator_wo_decay_model()

    # conjugate gradient recon with Tikhonov regularization
    app = sigpy.app.LinearLeastSquares(A, k_data, lamda=beta_reg)
    recon = app.run()

    return recon




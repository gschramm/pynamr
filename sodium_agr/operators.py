from __future__ import annotations

import numpy as np
import cupy as cp
import sigpy
import sigpy.mri

from types import ModuleType
from typing import Union, Sequence

# custom type indicating a numpy or cupy array
ndarray = Union[np.ndarray, cp.ndarray]


class ApodizedNUFFT(sigpy.linop.Linop):
    """TPI NUFFT operator with signal decay modeling

    Attributes
    ----------
    r : ndarray
        real decay ratio image
    tau : Sequence[float]
        list of tau values for each F operator
        the i-th element gives the readout time of the i-th F operator
        divided by the delta TE
    xp: ModuleType
        numpy or cupy module

    Note
    ----

    In the notation of the paper, this operator is a "vstack" of all
    A_i operators
    """

    def __init__(self, Flist: list[sigpy.linop.Linop], r: ndarray,
                 tau: Sequence[float]) -> None:
        """
        Parameters
        ----------
        Flist : list[sigpy.linop.Linop]
            list of Fourier (Forier Sense) operators
        r : ndarray
            real decay ratio image
        tau : Sequence[float]
            list of tau values for each F operator
            the i-th element gives the readout time of the i-th F operator
            divided by the delta TE
        """
        ishape = Flist[0].ishape
        oshape = (Flist[0].oshape[0], sum([x.oshape[1] for x in Flist]),
                  Flist[0].oshape[2])

        super().__init__(oshape, ishape)

        self._r = r
        self._tau = tau

        self._Flist = Flist
        self._num_time_bins = len(Flist)

        self._xp = sigpy.backend.get_array_module(self._r)

        # setup the index arrays that we need for the adjoint
        self._split_inds = []
        offset = 0
        for F in self._Flist:
            self._split_inds.append(np.arange(offset, offset + F.oshape[1]))
            offset += F.oshape[1]

    @property
    def r(self) -> ndarray:
        return self._r

    @r.setter
    def r(self, value: ndarray) -> None:
        self._r = value

    @property
    def tau(self) -> Sequence[float]:
        return self._tau

    @property
    def num_time_bins(self) -> int:
        return self._num_time_bins

    @property
    def xp(self) -> ModuleType:
        return self._xp

    def get_split_inds(self, i: int) -> ndarray:
        return self._split_inds[i]

    def get_Ai(self, i: int) -> sigpy.linop.Linop:
        return self._Flist[i] * sigpy.linop.Multiply(self.ishape, self._r**
                                                     self._tau[i])

    def _apply(self, input: ndarray) -> ndarray:
        y = cp.zeros(self.oshape, dtype=input.dtype)

        for i in range(len(self._Flist)):
            y[:, self._split_inds[i], :] = self.get_Ai(i)(input)

        return y

    def _adjoint(self, input: ndarray) -> ndarray:
        x = self._xp.zeros(self.ishape, dtype=input.dtype)

        for i in range(len(self._Flist)):
            x += self.get_Ai(i).H(input[:, self._split_inds[i], :])

        return x

    def _adjoint_linop(self) -> sigpy.linop.Linop:
        return _ApodizedNUFFTAdjoint(self)


class _ApodizedNUFFTAdjoint(sigpy.linop.Linop):

    def __init__(self, A: ApodizedNUFFT) -> None:

        oshape = A.ishape
        ishape = A.oshape

        super().__init__(oshape, ishape)

        self._A = A

    def _apply(self, input: ndarray) -> ndarray:
        return self._A._adjoint(input)

    def _adjoint_linop(self) -> sigpy.linop.Linop:
        return self._A


def projected_gradient_operator(xp: ModuleType,
                                prior_image: np.ndarray,
                                eta: float = 0.) -> sigpy.linop.Linop:
    """Projected gradient operator as defined in https://doi.org/10.1137/15M1047325.
       Gradient operator that return the component of a gradient that is orthogonal 
       to a joint gradient field (derived from a prior image)

    Parameters
    ----------
    xp: ModuleType
        numpy or cupy module
    prior_image : np.ndarray
        the prior image used to calcuate the joint gradient field for the projection

    Returns
    -------
    sigpy.linop.Linop
    """

    ishape = prior_image.shape

    # convert prior image to cupy GPU array if xp is cupy
    if xp.__name__ == 'cupy':
        prior_image = xp.asarray(prior_image)

    # normalized "normal" gradient operator
    G = (1 / np.sqrt(4 * len(ishape))) * sigpy.linop.FiniteDifference(
        ishape, axes=None)

    xi = G(prior_image)

    # normalize the real and imaginary part of the joint gradient field
    real_norm = xp.sqrt(xp.linalg.norm(xi.real, axis=0)**2 + eta**2)
    imag_norm = xp.sqrt(xp.linalg.norm(xi.imag, axis=0)**2 + eta**2)

    ir = xp.where(real_norm > 0)
    ii = xp.where(imag_norm > 0)

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

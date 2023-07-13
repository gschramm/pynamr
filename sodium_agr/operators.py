from __future__ import annotations

import numpy as np
import sigpy

from types import ModuleType


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

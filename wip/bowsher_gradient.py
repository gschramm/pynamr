from __future__ import annotations

import numpy as np

import math
from numba import njit, prange
import sigpy


@njit(parallel=True)
def nearest_neighbors_3d(img, s, ninds):
    """Calculate the n nearest neighbors for all voxels in a 3D array

    Parameters
    ----------
    img : 3d numpy array
      containing the image

    s : 3d binary (uint) numpy array
      containing the neighborhood definition.
      1 -> voxel is in neighborhood
      0 -> voxel is not in neighnorhood
      The dimensions of s have to be odd.

    ninds: 2d numpy array used for output
      of shape (xp.prod(img.shape), nnearest).
      ninds[i,:] contains the indicies of the nearest neighbors of voxel i

    Note
    ----
    All voxel indices are "flattened". It is assumed that the numpy arrays
    are in 'C' order.
    """

    nnearest = ninds.shape[1]

    offsets = np.array(s.shape) // 2
    maxdiff = img.max() - img.min() + 1

    d12 = img.shape[1] * img.shape[2]
    d2 = img.shape[2]

    for i0 in prange(img.shape[0]):
        for i1 in range(img.shape[1]):
            for i2 in range(img.shape[2]):

                absdiff = np.zeros(s.shape)
                val = img[i0, i1, i2]

                i_flattened = np.zeros(s.shape, dtype=ninds.dtype)

                for j0 in range(s.shape[0]):
                    for j1 in range(s.shape[1]):
                        for j2 in range(s.shape[2]):
                            tmp0 = i0 + j0 - offsets[0]
                            tmp1 = i1 + j1 - offsets[1]
                            tmp2 = i2 + j2 - offsets[2]

                            i_flattened[j0, j1, j2] = tmp0 * d12 + tmp1 * d2 + tmp2

                            if (
                                (tmp0 >= 0)
                                and (tmp0 < img.shape[0])
                                and (tmp1 >= 0)
                                and (tmp1 < img.shape[1])
                                and (tmp2 >= 0)
                                and (tmp2 < img.shape[2])
                                and s[j0, j1, j2] == 1
                            ):
                                absdiff[j0, j1, j2] = np.abs(
                                    img[tmp0, tmp1, tmp2] - val
                                )
                            else:
                                absdiff[j0, j1, j2] = maxdiff

                vox = i_flattened[offsets[0], offsets[1], offsets[2]]
                ninds[vox, :] = i_flattened.flatten()[
                    np.argsort(absdiff.flatten())[:nnearest]
                ]


# %%
class BowsherGradient(sigpy.linop.Linop):
    """Bowsher gradient operator using n nearest neighbors"""

    def __init__(
        self,
        structural_image: xp.ndarray,
        neighborhood: xp.ndarray,
        num_nearest: int,
        nearest_neighbor_inds: None | xp.ndarray = None,
    ) -> None:
        """init method

        Parameters
        ----------
        structural_image : xp.ndarray
            the structural image
        neighborhood : xp.ndarray
            the neighborhood definition image (1-> included, 0 -> excluded)
        num_nearest : int
            number of nearest neighbors used in gradient
        nearest_neighbor_inds : None | xp.ndarray, optional
            lookup table of nearest neighbor indices, by default None
            if None, gets calculated from structural_image and neighborhood
        """

        self._structural_image = structural_image
        self._neighborhood = neighborhood
        self._num_nearest = num_nearest

        if isinstance(structural_image, np.ndarray):
            self._xp = np
            from scipy.sparse import csc_matrix

            self._csc_matrix = csc_matrix
        else:
            import cupy as cp

            self._xp = cp
            from cupyx.scipy.sparse import csc_matrix

            self._csc_matrix = csc_matrix

        if nearest_neighbor_inds is None:
            self._nearest_neighbor_inds = np.zeros(
                (math.prod(structural_image.shape), num_nearest), dtype=int
            )

            if isinstance(structural_image, np.ndarray):
                nearest_neighbors_3d(
                    self._structural_image,
                    self._neighborhood,
                    self._nearest_neighbor_inds,
                )
            else:
                nearest_neighbors_3d(
                    self._xp.asnumpy(self._structural_image),
                    self._xp.asnumpy(self._neighborhood),
                    self._xp.asnumpy(self._nearest_neighbor_inds),
                )
                self._nearest_neighbor_inds = self._xp.asarray(
                    self._nearest_neighbor_inds
                )

        else:
            self._nearest_neighbor_inds = nearest_neighbor_inds

        num_voxels = structural_image.size
        tmp = self._xp.arange(num_voxels, dtype=float)
        diag = self._csc_matrix(
            (self._xp.full(num_voxels, -1, dtype=float), (tmp, tmp)),
            shape=(num_voxels, num_voxels),
        )

        self._sparse_fwd_diff_matrices = []

        for i in range(num_nearest):
            off_diag = self._csc_matrix(
                (
                    self._xp.full(num_voxels, 1, dtype=float),
                    (tmp, self._nearest_neighbor_inds[:, i]),
                ),
                shape=(num_voxels, num_voxels),
            )

            self._sparse_fwd_diff_matrices.append(diag + off_diag)

        super().__init__(
            oshape=(self._num_nearest,) + self._structural_image.shape,
            ishape=self._structural_image.shape,
        )

    @property
    def structural_image(self) -> xp.ndarray:
        return self._structural_image

    @property
    def neighborhood(self) -> xp.ndarray:
        return self._neighborhood

    @property
    def num_nearest(self) -> int:
        return self._num_nearest

    @property
    def nearest_neighbor_inds(self) -> xp.ndarray:
        return self._nearest_neighbor_inds

    @property
    def sparse_fwd_diff_matrices(self) -> list:
        return self._sparse_fwd_diff_matrices

    def _apply(self, x: xp.ndarray) -> xp.ndarray:
        y = self._xp.zeros(tuple(self.oshape), dtype=x.dtype)

        for i in range(self.num_nearest):
            y[i, ...] = self._xp.reshape(
                self.sparse_fwd_diff_matrices[i] @ x.ravel(), self.ishape
            )

        return y

    def _adjoint_linop(self) -> sigpy.linop.Linop:
        return AdjointBowsherGradient(
            self.structural_image,
            self.neighborhood,
            self.num_nearest,
            nearest_neighbor_inds=self.nearest_neighbor_inds,
        )


class AdjointBowsherGradient(sigpy.linop.Linop):
    """Adjoint of Bowsher gradient operator using n nearest neighbors"""

    def __init__(
        self,
        structural_image: xp.ndarray,
        neighborhood: xp.ndarray,
        num_nearest: int,
        nearest_neighbor_inds: None | xp.ndarray = None,
    ) -> None:
        """init method

        Parameters
        ----------
        structural_image : xp.ndarray
            the structural image
        neighborhood : xp.ndarray
            the neighborhood definition image (1-> included, 0 -> excluded)
        num_nearest : int
            number of nearest neighbors used in gradient
        nearest_neighbor_inds : None | xp.ndarray, optional
            lookup table of nearest neighbor indices, by default None
            if None, gets calculated from structural_image and neighborhood
        """

        self._structural_image = structural_image
        self._neighborhood = neighborhood
        self._num_nearest = num_nearest

        if isinstance(structural_image, np.ndarray):
            self._xp = np
            from scipy.sparse import csc_matrix

            self._csc_matrix = csc_matrix
        else:
            import cupy as cp

            self._xp = cp
            from cupyx.scipy.sparse import csc_matrix

            self._csc_matrix = csc_matrix

        if nearest_neighbor_inds is None:
            self._nearest_neighbor_inds = np.zeros(
                (math.prod(structural_image.shape), num_nearest), dtype=int
            )

            if isinstance(structural_image, np.ndarray):
                nearest_neighbors_3d(
                    self._structural_image,
                    self._neighborhood,
                    self._nearest_neighbor_inds,
                )
            else:
                nearest_neighbors_3d(
                    self._xp.asnumpy(self._structural_image),
                    self._xp.asnumpy(self._neighborhood),
                    self._xp.asnumpy(self._nearest_neighbor_inds),
                )
                self._nearest_neighbor_inds = self._xp.asarray(
                    self._nearest_neighbor_inds
                )

        else:
            self._nearest_neighbor_inds = nearest_neighbor_inds

        num_voxels = structural_image.size
        tmp = self._xp.arange(num_voxels, dtype=float)
        diag = self._csc_matrix(
            (self._xp.full(num_voxels, -1, dtype=float), (tmp, tmp)),
            shape=(num_voxels, num_voxels),
        )

        self._sparse_fwd_diff_matrices = []

        for i in range(num_nearest):
            off_diag = self._csc_matrix(
                (
                    self._xp.full(num_voxels, 1, dtype=float),
                    (tmp, self._nearest_neighbor_inds[:, i]),
                ),
                shape=(num_voxels, num_voxels),
            ).T

            self._sparse_fwd_diff_matrices.append(diag + off_diag)

        super().__init__(
            ishape=(self._num_nearest,) + self._structural_image.shape,
            oshape=self._structural_image.shape,
        )

    @property
    def structural_image(self) -> xp.ndarray:
        return self._structural_image

    @property
    def neighborhood(self) -> xp.ndarray:
        return self._neighborhood

    @property
    def num_nearest(self) -> int:
        return self._num_nearest

    @property
    def nearest_neighbor_inds(self) -> xp.ndarray:
        return self._nearest_neighbor_inds

    @property
    def sparse_fwd_diff_matrices(self) -> list:
        return self._sparse_fwd_diff_matrices

    def _apply(self, y: xp.ndarray) -> xp.ndarray:
        x = self._xp.zeros(tuple(self.oshape), dtype=y.dtype)

        for i in range(self.num_nearest):
            x += self._xp.reshape(
                self.sparse_fwd_diff_matrices[i] @ y[i, ...].ravel(), self.oshape
            )

        return x

    def _adjoint_linop(self) -> sigpy.linop.Linop:
        return BowsherGradient(
            self.structural_image,
            self.neighborhood,
            self.num_nearest,
            nearest_neighbor_inds=self.nearest_neighbor_inds,
        )


# %%
if __name__ == "__main__":
    import cupy as cp

    # import numpy as cp

    cp.random.seed(0)
    shape = (128, 127, 126)
    aimg = cp.random.rand(*shape)

    s = cp.ones((5, 5, 5), dtype=cp.uint8)
    s[0, 0, 0] = 0
    s[-1, 0, 0] = 0
    s[0, -1, 0] = 0
    s[0, 0, -1] = 0
    s[-1, -1, 0] = 0
    s[-1, 0, -1] = 0
    s[0, -1, -1] = 0
    s[-1, -1, -1] = 0

    s[s.shape[0] // 2, s.shape[1] // 2, s.shape[2] // 2] = 0

    bow_grad = BowsherGradient(aimg, s, 13)

    # %%
    x = cp.random.rand(*bow_grad.ishape) + 1.0j ** cp.random.rand(*bow_grad.ishape)
    y = cp.random.rand(*bow_grad.oshape) + 1.0j ** cp.random.rand(*bow_grad.oshape)

    x_fwd = bow_grad(x)
    y_adj = bow_grad.H(y)
    assert cp.isclose(cp.sum(x_fwd * y), cp.sum(x * y_adj))

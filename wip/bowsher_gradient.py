from __future__ import annotations

import numpy as np
from scipy.sparse import coo_array
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
      of shape (np.prod(img.shape), nnearest).
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
    def __init__(
        self,
        structural_image: np.ndarray,
        neighborhood: np.ndarray,
        num_nearest: int,
        nearest_neighbor_inds: None | np.ndarray = None,
    ) -> None:
        self._structural_image = structural_image
        self._neighborhood = neighborhood
        self._num_nearest = num_nearest

        if nearest_neighbor_inds is None:
            self._nearest_neighbor_inds = np.zeros(
                (np.prod(structural_image.shape), num_nearest), dtype=int
            )

            nearest_neighbors_3d(
                self._structural_image, self._neighborhood, self._nearest_neighbor_inds
            )
        else:
            self._nearest_neighbor_inds = nearest_neighbor_inds

        num_voxels = structural_image.size
        tmp = np.arange(num_voxels)
        diag = coo_array(
            (np.full(num_voxels, -1, dtype=int), (tmp, tmp)),
            shape=(num_voxels, num_voxels),
        )

        self._sparse_fwd_diff_matrices = []

        for i in range(num_nearest):
            off_diag = coo_array(
                (
                    np.full(num_voxels, 1, dtype=int),
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
    def structural_image(self) -> np.ndarray:
        return self._structural_image

    @property
    def neighborhood(self) -> np.ndarray:
        return self._neighborhood

    @property
    def num_nearest(self) -> int:
        return self._num_nearest

    @property
    def nearest_neighbor_inds(self) -> np.ndarray:
        return self._nearest_neighbor_inds

    @property
    def sparse_fwd_diff_matrices(self) -> list:
        return self._sparse_fwd_diff_matrices

    def _apply(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros(tuple(self.oshape), dtype = x.dtype)

        for i in range(self.num_nearest):
            y[i, ...] = np.reshape(
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
    def __init__(
        self,
        structural_image: np.ndarray,
        neighborhood: np.ndarray,
        num_nearest: int,
        nearest_neighbor_inds: None | np.ndarray = None,
    ) -> None:
        self._structural_image = structural_image
        self._neighborhood = neighborhood
        self._num_nearest = num_nearest

        if nearest_neighbor_inds is None:
            self._nearest_neighbor_inds = np.zeros(
                (np.prod(structural_image.shape), num_nearest), dtype=int
            )

            nearest_neighbors_3d(
                self._structural_image, self._neighborhood, self._nearest_neighbor_inds
            )
        else:
            self._nearest_neighbor_inds = nearest_neighbor_inds

        num_voxels = structural_image.size
        tmp = np.arange(num_voxels)
        diag = coo_array(
            (np.full(num_voxels, -1, dtype=int), (tmp, tmp)),
            shape=(num_voxels, num_voxels),
        )

        self._sparse_fwd_diff_matrices = []

        for i in range(num_nearest):
            off_diag = coo_array(
                (
                    np.full(num_voxels, 1, dtype=int),
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
    def structural_image(self) -> np.ndarray:
        return self._structural_image

    @property
    def neighborhood(self) -> np.ndarray:
        return self._neighborhood

    @property
    def num_nearest(self) -> int:
        return self._num_nearest

    @property
    def nearest_neighbor_inds(self) -> np.ndarray:
        return self._nearest_neighbor_inds

    @property
    def sparse_fwd_diff_matrices(self) -> list:
        return self._sparse_fwd_diff_matrices

    def _apply(self, y: np.ndarray) -> np.ndarray:
        x = np.zeros(tuple(self.oshape), dtype = y.dtype)

        for i in range(self.num_nearest):
            x += np.reshape(
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
np.random.seed(0)
shape = (128, 126, 127)
aimg = np.random.rand(*shape)

s = np.ones((3, 3, 3), dtype=np.uint8)
s[0, 0, 0] = 0
s[-1, 0, 0] = 0
s[0, -1, 0] = 0
s[0, 0, -1] = 0
s[-1, -1, 0] = 0
s[-1, 0, -1] = 0
s[0, -1, -1] = 0
s[-1, -1, -1] = 0

s[s.shape[0] // 2, s.shape[1] // 2, s.shape[2] // 2] = 0

bow_grad = BowsherGradient(aimg, s, 4)


# %%
x = np.random.rand(*bow_grad.ishape) + 1.0j**np.random.rand(*bow_grad.ishape)
y = np.random.rand(*bow_grad.oshape) + 1.0j**np.random.rand(*bow_grad.oshape)

x_fwd = bow_grad(x)
y_adj = bow_grad.H(y)

assert np.isclose(np.sum(x_fwd*y), np.sum(x*y_adj))

# nn = img.size
# tmp = np.arange(nn)
# print('setting up sparse matrices')
# a = coo_array((np.full(nn, -1, dtype=int), (tmp, tmp)), shape=(nn, nn))
#
# diff_matrices = []
#
# for i in range(ninds.shape[1]):
#    b = coo_array((np.full(nn, 1, dtype=int), (tmp, ninds[:, i])), shape=(nn, nn))
#    diff_matrices.append(a + b)
#
# print('applying sparse matrices')
# x0 = np.reshape(diff_matrices[0] @ img.flatten(), shape)
# x1 = np.reshape(diff_matrices[1] @ img.flatten(), shape)
# x2 = np.reshape(diff_matrices[2] @ img.flatten(), shape)
#
# print('done')

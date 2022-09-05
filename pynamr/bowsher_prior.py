from typing import Protocol

import numpy as np
from numba import njit, jit, prange

from .protocols import DifferentiableFunction


@njit(parallel=True)
def next_neighbors(shape: tuple[int, int], ninds: np.ndarray) -> None:
    """ Calculate the 2/3 next neighbors for all voxels in a 2/3D array.
        Usefule for standard fwd finite differences (without structural information).

        Parameters
        ----------
        shape : tuple
          shape of the 2D/3D image

        ninds: 2d numpy array used for output
          of shape (np.prod(img.shape), 2/3).
          ninds[i,:] contains the indicies of the nearest neighbors of voxel i

        Note
        ----
        All voxel indices are "flattened". It is assumed that the numpy arrays
        are in 'C' order.
    """

    if len(shape) == 2:
        d1 = shape[1]

        for i0 in prange(shape[0]):
            for i1 in range(shape[1]):
                j = i0 * d1 + i1
                ninds[j, 0] = i0 * d1 + ((i1 + 1) % shape[1])
                ninds[j, 1] = ((i0 + 1) % shape[0]) * d1 + i1

    elif len(shape) == 3:
        d12 = shape[1] * shape[2]
        d2 = shape[2]

        for i0 in prange(shape[0]):
            for i1 in range(shape[1]):
                for i2 in range(shape[2]):
                    j = i0 * d12 + i1 * d2 + i2
                    ninds[j, 0] = i0 * d12 + i1 * d2 + ((i2 + 1) % shape[2])
                    ninds[j, 1] = i0 * d12 + ((i1 + 1) % shape[1]) * d2 + i2
                    ninds[j, 2] = ((i0 + 1) % shape[0]) * d12 + i1 * d2 + i2


@njit(parallel=True)
def nearest_neighbors_3d(img: np.ndarray, s: np.ndarray, nnearest: int,
                         ninds: np.ndarray) -> None:
    """ Calculate the n nearest neighbors for all voxels in a 3D array

        Parameters
        ----------
        img : 3d numpy array
          containing the image

        s : 3d binary (uint) numpy array
          containing the neighborhood definition.
          1 -> voxel is in neighborhood
          0 -> voxel is not in neighnorhood
          The dimensions of s have to be odd.

        nnearest : uint
          number of nearest neighbors
          Make sure that nnearest is small enough for the "corner" voxels given s.

        ninds: 2d numpy array used for output
          of shape (np.prod(img.shape), nnearest).
          ninds[i,:] contains the indicies of the nearest neighbors of voxel i

        Note
        ----
        All voxel indices are "flattened". It is assumed that the numpy arrays
        are in 'C' order.
    """
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

                            i_flattened[j0, j1,
                                        j2] = tmp0 * d12 + tmp1 * d2 + tmp2

                            if ((tmp0 >= 0) and (tmp0 < img.shape[0])
                                    and (tmp1 >= 0) and (tmp1 < img.shape[1])
                                    and (tmp2 >= 0) and (tmp2 < img.shape[2])
                                    and s[j0, j1, j2] == 1):
                                absdiff[j0, j1,
                                        j2] = np.abs(img[tmp0, tmp1, tmp2] -
                                                     val)
                            else:
                                absdiff[j0, j1, j2] = maxdiff

                vox = i_flattened[offsets[0], offsets[1], offsets[2]]
                ninds[vox, :] = i_flattened.flatten()[np.argsort(
                    absdiff.flatten())[:nnearest]]


def nearest_neighbors_2d(img: np.ndarray, s: np.ndarray, nnearest: int,
                         ninds: np.ndarray):
    """ Calculate the n nearest neighbors for all voxels in a 2D array

        Parameters
        ----------
        img : 2d numpy array
          containing the image

        s : 2d binary (uint) numpy array
          containing the neighborhood definition.
          1 -> voxel is in neighborhood
          0 -> voxel is not in neighnorhood
          The dimensions of s have to be odd.

        nnearest : uint
          number of nearest neighbors
          Make sure that nnearest is small enough for the "corner" voxels given s.

        ninds: 2d numpy array used for output
          of shape (np.prod(img.shape), nnearest).
          ninds[i,:] contains the indicies of the nearest neighbors of voxel i

        Note
        ----
        (1) All voxel indices are "flattened". It is assumed that the numpy arrays
        are in 'C' order.
        (2) the input 2D arrays are converted to 3D arrays and passed to nearest_neighbors_3d
    """
    z = np.zeros(s.shape, dtype=s.dtype)
    s2 = np.array([z, s, z])

    nearest_neighbors_3d(np.expand_dims(img, 0), s2, nnearest, ninds)


def nearest_neighbors(img: np.ndarray, s: np.ndarray, nnearest: int,
                      ninds: np.ndarray) -> None:
    """ Calculate the n nearest neighbors for all voxels in a 2D or 3D array

        Parameters
        ----------
        img : 2d or 3d numpy array
          containing the image

        s : 2d or 3d binary (uint) numpy array
          containing the neighborhood definition.
          1 -> voxel is in neighborhood
          0 -> voxel is not in neighnorhood
          The dimensions of s have to be odd.

        nnearest : uint
          number of nearest neighbors
          Make sure that nnearest is small enough for the "corner" voxels given s.

        ninds: 2d numpy array used for output
          of shape (np.prod(img.shape), nnearest).
          ninds[i,:] contains the indicies of the nearest neighbors of voxel i

        Note
        ----
        (1) All voxel indices are "flattened". It is assumed that the numpy arrays
        are in 'C' order.
        (2) depending on the number of dimension of the input image,
            nearest_neighbors_2d or nearest_neighbors_3d is called.
    """
    if img.ndim == 2:
        nearest_neighbors_2d(img, s, nnearest, ninds)
    elif img.ndim == 3:
        nearest_neighbors_3d(img, s, nnearest, ninds)
    else:
        raise ValueError("input image must be 2d or 3d")


def is_nearest_neighbor_of(ninds: np.ndarray) -> np.ndarray:
    """ Given an 2d array of nearest neighbors for each voxel, calculate
        for which voxels the voxel is a nearest neighbor

        Parameters
        ----------
        ninds : 2d numpy array
          ninds[i,:] contains the nearest neighbors of voxel i


        Returns
        -------
        A 2d array of shape (2, ninds.flatten().shape[0]).
        output_array[0,:] contains a voxel number and j
        output_array[0,:] contains a voxel for which j is a nearest neighbor
    """
    sinds = np.argsort(ninds.flatten())
    ninds_adjoint = np.zeros((2, sinds.shape[0]), dtype=ninds.dtype)
    ninds_adjoint[0, :] = ninds.flatten()[sinds]
    ninds_adjoint[1, :] = sinds // ninds.shape[1]

    return ninds_adjoint


@njit(parallel=True)
def bowsher_cost(img: np.ndarray, ninds: np.ndarray) -> float:
    img_shape = img.shape
    img = img.flatten()
    cost = 0.

    for i in prange(ninds.shape[0]):
        for j in range(ninds.shape[1]):
            cost += 0.5 * (img[i] - img[ninds[i, j]])**2

    img = img.reshape(img_shape)

    return cost


@njit(parallel=True)
def bowsher_grad(img: np.ndarray, ninds: np.ndarray,
                 ninds_adj: np.ndarray) -> np.ndarray:
    img_shape = img.shape
    img = img.flatten()
    grad = np.zeros(img.shape, dtype=img.dtype)

    counter = 0

    for i in range(ninds.shape[0]):
        # first term
        for j in range(ninds.shape[1]):
            grad[i] += (img[i] - img[ninds[i, j]])

        # 2nd term
        while (counter < ninds_adj.shape[1]) and (ninds_adj[0, counter] == i):
            grad[i] += (img[i] - img[ninds_adj[1, counter]])
            counter += 1

    img = img.reshape(img_shape)
    grad = grad.reshape(img_shape)

    return grad

class BowsherLoss(DifferentiableFunction):

    def __init__(self, ninds: np.ndarray, ninds_adj: np.ndarray) -> None:
        self.ninds = ninds
        self.ninds_adj = ninds_adj

    def __call__(self, img: np.ndarray) -> float:
        return bowsher_cost(img, self.ninds)

    def grad(self, img: np.ndarray) -> np.ndarray:
        return bowsher_grad(img, self.ninds, self.ninds_adj)

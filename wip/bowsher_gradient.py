import numpy as np
from numba import njit, prange


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


@njit(parallel=True)
def bowsher_gradient(img, ninds):
    img_shape = img.shape
    img = img.flatten()

    grad = np.zeros(img.shape + (ninds.shape[1],), dtype=img.dtype)

    for i in prange(ninds.shape[0]):
        for j in range(ninds.shape[1]):
            grad[i, j] = img[ninds[i, j]] - img[i]

    img = img.reshape(img_shape)

    grad = grad.reshape(img_shape + (ninds.shape[1],))

    return grad


# %%
np.random.seed(0)
shape = (128, 120, 111)
aimg = np.random.rand(*shape)

s = np.ones((3, 3, 3), dtype=np.uint8)
s[s.shape[0] // 2, s.shape[1] // 2, s.shape[2] // 2] = 0

ninds = np.zeros((np.prod(shape), 5), dtype=np.int64)

nearest_neighbors_3d(aimg, s, ninds)

# %%
img = np.random.rand(*shape)
g = bowsher_gradient(img, ninds)

# %%
from scipy.sparse import coo_array

nn = img.size
tmp = np.arange(nn)

a = coo_array((np.full(nn, -1, dtype=int), (tmp, tmp)), shape=(nn, nn))

diff_matrices = []

for i in range(ninds.shape[1]):
    b = coo_array((np.full(nn, 1, dtype=int), (tmp, ninds[:, i])), shape=(nn, nn))
    diff_matrices.append(a + b)

x0 = np.reshape(diff_matrices[0] @ img.flatten(), shape)
x1 = np.reshape(diff_matrices[1] @ img.flatten(), shape)
x2 = np.reshape(diff_matrices[2] @ img.flatten(), shape)

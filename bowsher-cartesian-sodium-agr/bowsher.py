import numpy as np
from numba import njit


@njit()
def rel_diff(a, b):
    if (a + b) != 0:
        f = ((a - b)**2) / (a + b)
    else:
        f = 0

    return f


@njit()
def grad0_rel_diff(a, b):
    if (a + b) != 0:
        g = (a**2 + 2 * a * b - 3 * (b**2)) / ((a + b)**2)
    else:
        g = -1

    return g


@njit()
def grad1_rel_diff(a, b):
    if (a + b) != 0:
        g = (b**2 + 2 * a * b - 3 * (a**2)) / ((a + b)**2)
    else:
        g = -1

    return g


@njit()
def bowsher_prior_cost(img, ninds, method):
    img_shape = img.shape
    img = img.flatten()
    cost = 0.

    if method == 0:
        for i in range(ninds.shape[0]):
            for j in range(ninds.shape[1]):
                cost += (img[i] - img[ninds[i, j]])**2
    elif method == 1:
        for i in range(ninds.shape[0]):
            for j in range(ninds.shape[1]):
                cost += rel_diff(img[i], img[ninds[i, j]])

    img = img.reshape(img_shape)

    return cost


@njit(parallel=True)
def bowsher_prior_grad(img, ninds, ninds2, method, asym=0):
    img_shape = img.shape
    img = img.flatten()
    grad = np.zeros(img.shape, dtype=img.dtype)

    counter = 0

    if method == 0:
        for i in range(ninds.shape[0]):
            # first term
            for j in range(ninds.shape[1]):
                grad[i] += 2 * (img[i] - img[ninds[i, j]])

            # 2nd term
            if asym == 0:
                while (counter < ninds2.shape[1]) and (ninds2[0, counter]
                                                       == i):
                    grad[i] += -2 * (img[ninds2[1, counter]] - img[i])
                    counter += 1
    elif method == 1:
        for i in range(ninds.shape[0]):
            # first term
            for j in range(ninds.shape[1]):
                grad[i] += grad0_rel_diff(img[i], img[ninds[i, j]])

            # 2nd term
            if asym == 0:
                while (counter < ninds2.shape[1]) and (ninds2[0, counter]
                                                       == i):
                    grad[i] += grad1_rel_diff(img[ninds2[1, counter]], img[i])
                    counter += 1

    img = img.reshape(img_shape)
    grad = grad.reshape(img_shape)

    return grad

import numpy as np


def rod_phantom(n=512,
                r=0.9,
                r_rod=0.08,
                nrods=5,
                rod_contrast=None,
                rod_gam=None,
                dt=5.):

    if rod_contrast is None:
        rod_contrast = np.linspace(0.1, 1.2, nrods)
    if rod_gam is None and dt>0.:
        # T2* decay between TE1 and TE2 with realistic brain T2* values
        rod_gam = np.exp(-dt/np.array([2, 5, 10, 20, 40])) #ms

    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x, indexing='ij')

    R = np.sqrt((X - 0)**2 + (Y - 0)**2)

    x = np.zeros((n, n))
    gam = np.ones((n, n))

    x[R < r] = 1
    gam[R < r] = 0.7

    for i, ox in enumerate(
            np.linspace(-0.8 * r / np.sqrt(2), 0.8 * r / np.sqrt(2), nrods)):
        for j, oy in enumerate(
                np.linspace(-0.8 * r / np.sqrt(2), 0.8 * r / np.sqrt(2),
                            nrods)):
            R = np.sqrt((X - ox)**2 + (Y - oy)**2)
            x[R < r_rod] = rod_contrast[i]
            gam[R < r_rod] = rod_gam[j]

    # repeat 2D arrays into 3D array
    x = np.repeat(x[:, :, np.newaxis], n, axis=2)
    gam = np.repeat(gam[:, :, np.newaxis], n, axis=2)

    # delete first / last slices
    x[:, :, :(n // 8)] = 0
    x[:, :, ((7 * n) // 8):] = 0

    gam[:, :, :(n // 8)] = 1
    gam[:, :, ((7 * n) // 8):] = 1

    return x, gam

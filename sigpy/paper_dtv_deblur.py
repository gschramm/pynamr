"""minimal script that shows how to solve L2square data fidelity + anatomical DTV prior"""

import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndimage
import sigpy

import matplotlib.pyplot as plt

from utils_sigpy import projected_gradient_operator


class GaussConv(sigpy.linop.Linop):

    def __init__(self, img_shape, sigma):
        super().__init__(img_shape, img_shape)
        self._sigma = sigma

    def _apply(self, input):
        return ndimage.gaussian_filter(input, self._sigma)

    def _adjoint_linop(self):
        return self


#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

n = 128
img_shape = (n, n)
sigma = 3.
noise_level = 0.2
max_iter = 2000

betas = np.round(np.logspace(np.log10(0.01), np.log10(3), 4), 2)

seed = 0
#------------------------------------------------------------

cp.random.seed(seed)

# Gaussian blurring "data" operator
A = GaussConv(img_shape, sigma)

true_image = cp.zeros(img_shape)
true_image[(1 * (n // 4)):(3 * (n // 4)), (1 * (n // 4)):(3 * (n // 4))] = 1.
true_image[(3 * (n // 8)):(5 * (n // 8)), (3 * (n // 8)):(5 * (n // 8))] = 1.5
true_image[(15 * (n // 32)):(17 * (n // 32)),
           (15 * (n // 32)):(17 * (n // 32))] = 1.0

prior_image = (true_image.max() -
               true_image)**0.5 + 0.02 * cp.random.randn(*img_shape)

true_image[(21 * (n // 32)):(23 * (n // 32)),
           (21 * (n // 32)):(23 * (n // 32))] = 1.5

prior_image[(9 * (n // 32)):(11 * (n // 32)),
            (9 * (n // 32)):(11 * (n // 32))] = 0

# projected Gradient operator
G = sigpy.linop.Gradient(img_shape)
PG = projected_gradient_operator(img_shape, prior_image, eta=0.04)

noise_free_data = A(true_image)
data = noise_free_data.copy() + noise_level * cp.random.randn(*img_shape)

tv_recons = []
dtv_recons = []

for i, beta in enumerate(betas):
    proxg = sigpy.prox.L1Reg(PG.oshape, lamda=beta)

    alg = sigpy.app.LinearLeastSquares(
        A,
        data,
        x=None,
        G=G,
        proxg=proxg,
        max_iter=max_iter,
    )

    res = alg.run()

    tv_recons.append(res)

    alg2 = sigpy.app.LinearLeastSquares(
        A,
        data,
        x=None,
        G=PG,
        proxg=proxg,
        max_iter=max_iter,
    )

    res2 = alg2.run()

    dtv_recons.append(res2)

#------------------------------------------------------------------------

ims = dict(vmin=0, vmax=1.7, cmap='gray')

nrows = 3
ncols = len(betas)

fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
ax[0, 0].imshow(cp.asnumpy(true_image), **ims)
ax[0, 0].set_title('ground truth')
ax[0, 1].imshow(cp.asnumpy(noise_free_data), **ims)
ax[0, 1].set_title('blurred ground truth')
ax[0, 2].imshow(cp.asnumpy(data), **ims)
ax[0, 2].set_title('blurred ground truth w. noise')
ax[0, 3].imshow(cp.asnumpy(prior_image), **ims)
ax[0, 3].set_title('prior image')

for i, beta in enumerate(betas):
    ax[1, i].imshow(cp.asnumpy(tv_recons[i]), **ims)
    ax[1, i].set_title(f'recon. w. TV weight = {beta}')
    ax[2, i].imshow(cp.asnumpy(dtv_recons[i]), **ims)
    ax[2, i].set_title(f'recon. w. dTV weight = {beta}')

for axx in ax.ravel():
    axx.set_axis_off()

fig.tight_layout()
fig.show()
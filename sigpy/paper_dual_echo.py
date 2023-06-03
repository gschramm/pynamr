import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

pdir = Path('/data/sodium_mr/NYU/CSF31_raw/recons_128')

r1 = np.abs(
    np.load(pdir / 'recon_echo_1_no_decay_model_L2_1.0E-01_2000.npz')['x'])
r2 = np.abs(
    np.load(pdir / 'recon_echo_2_no_decay_model_L2_1.0E-01_2000.npz')['x'])

ratio = r2 / (r1 + 0.01)

ims = dict(origin='upper')

sl = 64

fig, ax = plt.subplots(1, 3, figsize=(2.5 * 2.5, 2.5))
im0 = ax[0].imshow(r1[sl, :, :].T, vmin=0, vmax=1.3, cmap='gray', **ims)
im1 = ax[1].imshow(r2[sl, :, :].T, vmin=0, vmax=1.3, cmap='gray', **ims)
im2 = ax[2].imshow(ratio[sl, :, :].T, vmin=0.2, vmax=1., cmap='magma', **ims)

fig.colorbar(im0, ax=ax[0], location='bottom', fraction=0.04, pad=0.01)
fig.colorbar(im1, ax=ax[1], location='bottom', fraction=0.04, pad=0.01)
fig.colorbar(im2, ax=ax[2], location='bottom', fraction=0.04, pad=0.01)

ax[0].set_title('TE_1 = 0.5 ms')
ax[1].set_title('TE_2 = 5 ms')
ax[2].set_title('echo 2 / echo 1')

for axx in ax.ravel():
    axx.set_axis_off()

fig.tight_layout()
fig.show()
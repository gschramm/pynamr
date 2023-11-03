import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pymirc.viewer as pv

matrix_size = 64

if matrix_size == 128:
    betas = [0., 64., 512., 1024., 2048., 4096.]
else:
    betas = [0., 32., 64., 128., 256., 512., 1024.]

decay_suffix = '_no_decay_model'

odir16 = Path(f'/data/sodium_mr/20230316_MR3_GS_QED/pfiles/g16/grad_test_recons_{matrix_size}')
odir32 = Path(f'/data/sodium_mr/20230316_MR3_GS_QED/pfiles/g32/grad_test_recons_{matrix_size}')

recons_16 = np.zeros((len(betas), matrix_size, matrix_size, matrix_size), dtype = complex)
recons_32 = np.zeros((len(betas), matrix_size, matrix_size, matrix_size), dtype = complex)
recons_32_clipped = np.zeros((len(betas), matrix_size, matrix_size, matrix_size), dtype = complex)

for ib, beta in enumerate(betas):
    # convert flat pseudo complex array to complex
    recons_16[ib,...] = np.load(odir16 / f'recon_quad_prior_beta_{beta:.1E}{decay_suffix}.npy')
    recons_32[ib,...] = np.flip(np.swapaxes(np.load(odir32 / f'recon_quad_prior_beta_{beta:.1E}{decay_suffix}.npy'), 0, 1), 0)
    recons_32_clipped[ib,...] = np.flip(np.swapaxes(np.load(odir32 / f'recon_quad_prior_beta_{beta:.1E}{decay_suffix}_clipped_kmax.npy'), 0, 1), 0)


#----------------------------------
ims = dict(vmax = 9 * (128/matrix_size)**1.5, cmap = 'Greys_r')
sl_y = 32
sl_z = 40

plt.style.use('dark_background')

fig, ax = plt.subplots(6, len(betas), figsize = (1.5*len(betas),1.5*6))
for ib, beta in enumerate(betas):
    ax[0,ib].imshow(np.abs(recons_16[ib,:,:,sl_z]), **ims)
    ax[1,ib].imshow(np.abs(recons_32_clipped[ib,:,:,sl_z]), **ims)
    ax[2,ib].imshow(np.abs(recons_32[ib,:,:,sl_z]), **ims)

    ax[3,ib].imshow(np.abs(recons_16[ib,:,sl_y,:]).T, origin = 'lower', **ims)
    ax[4,ib].imshow(np.abs(recons_32_clipped[ib,:,sl_y,:]).T, origin = 'lower', **ims)
    ax[5,ib].imshow(np.abs(recons_32[ib,:,sl_y,:]).T, origin = 'lower', **ims)

    ax[0, ib].set_title(f'b = {int(beta)}', fontsize = 'medium')
    ax[3, ib].set_title(f'b = {int(beta)}', fontsize = 'medium')
for axx in ax.ravel():
    axx.set_axis_off()
fig.tight_layout()
fig.show()


#vi = pv.ThreeAxisViewer([np.abs(recons_16), np.abs(recons_32), np.abs(recons_32_clipped)], imshow_kwargs = ims)
import h5py
import os
import numpy as np

import matplotlib
import matplotlib.lines as lines
import matplotlib.pyplot as py

py.rc('image', cmap='gray')

py.rcParams['text.usetex'] = True
py.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']
py.rcParams['font.family'] = 'sans-serif'
py.rcParams['font.sans-serif'] = 'Computer Modern Sans Serif'
py.rcParams['axes.titlesize'] =  'medium'
py.rcParams['font.size'] =  12

vmax = 1.2

fnames = [
'T2star_recon_short_-1.0__T2star_recon_long_-1.0__beta_0.5__sm_fwhm_1.5__noise_level_0.2__n_256__niter_500__method_0__T2star_csf_short_50__T2star_csf_long_50__T2star_gm_short_8__T2star_gm_long_15__T2star_wm_short_9__T2star_wm_long_18.h5',
'T2star_recon_short_-1.0__T2star_recon_long_-1.0__beta_2.0__sm_fwhm_1.5__noise_level_0.2__n_256__niter_500__method_0__T2star_csf_short_50__T2star_csf_long_50__T2star_gm_short_8__T2star_gm_long_15__T2star_wm_short_9__T2star_wm_long_18.h5',
'T2star_recon_short_-1.0__T2star_recon_long_-1.0__beta_10.0__sm_fwhm_1.5__noise_level_0.2__n_256__niter_500__method_0__T2star_csf_short_50__T2star_csf_long_50__T2star_gm_short_8__T2star_gm_long_15__T2star_wm_short_9__T2star_wm_long_18.h5',
'T2star_recon_short_8.0__T2star_recon_long_15.0__beta_0.5__sm_fwhm_1.5__noise_level_0.2__n_256__niter_500__method_0__T2star_csf_short_50__T2star_csf_long_50__T2star_gm_short_8__T2star_gm_long_15__T2star_wm_short_9__T2star_wm_long_18.h5',
'T2star_recon_short_8.0__T2star_recon_long_15.0__beta_2.0__sm_fwhm_1.5__noise_level_0.2__n_256__niter_500__method_0__T2star_csf_short_50__T2star_csf_long_50__T2star_gm_short_8__T2star_gm_long_15__T2star_wm_short_9__T2star_wm_long_18.h5',
'T2star_recon_short_8.0__T2star_recon_long_15.0__beta_10.0__sm_fwhm_1.5__noise_level_0.2__n_256__niter_500__method_0__T2star_csf_short_50__T2star_csf_long_50__T2star_gm_short_8__T2star_gm_long_15__T2star_wm_short_9__T2star_wm_long_18.h5']

bow_recons   = np.zeros((6,256,256))
noreg_recons = np.zeros((6,256,256))

for i, fname in enumerate(fnames):
  with h5py.File(os.path.join('./data/recons/mic20',fname), 'r') as h5data:
    bow_recons[i,...]   = np.linalg.norm(h5data['images/bow_recon'][:],axis=-1)
    noreg_recons[i,...] = np.linalg.norm(h5data['images/noreg_recon'][:],axis=-1)

    if i == 0:
      ifft_recon = np.linalg.norm(h5data['images/ifft_recon'][:],axis=-1)
      gt         = np.linalg.norm(h5data['images/ground_truth'][:],axis=-1)

fig3, ax3 = py.subplots(2,5, figsize = (12.,5), squeeze = False)
ax3[0,0].imshow(noreg_recons[0,...], vmax = vmax)
ax3[1,0].imshow(noreg_recons[3,...], vmax = vmax)

ax3[0,0].set_ylabel('perfect fwd model')
ax3[1,0].set_ylabel('simplified fwd model')

ax3[0,1].imshow(bow_recons[0,...], vmax = vmax)
ax3[0,2].imshow(bow_recons[1,...], vmax = vmax)
ax3[0,3].imshow(bow_recons[2,...], vmax = vmax)
ax3[1,1].imshow(bow_recons[3,...], vmax = vmax)
ax3[1,2].imshow(bow_recons[4,...], vmax = vmax)
ax3[1,3].imshow(bow_recons[5,...], vmax = vmax)

ax3[0,4].imshow(gt, vmax = vmax)
ax3[1,4].imshow(ifft_recon, vmax = vmax)

ax3[0,0].set_title(r'$\beta = 0$')
ax3[0,1].set_title(r'$\beta = 0.2$')
ax3[0,2].set_title(r'$\beta = 2$')
ax3[0,3].set_title(r'$\beta = 10$')
ax3[0,4].set_title(r'ground truth')
ax3[1,4].set_title(r'IFFT Han filtered data')

for axx in ax3.flatten(): 
  axx.set_xticklabels([])
  axx.set_yticklabels([])
  axx.set_xticks([])
  axx.set_yticks([])

ax3[-1,-1].set_axis_off()

l1 = lines.Line2D([0.8, 0.8], [0, 1.], transform=fig3.transFigure, figure=fig3, linewidth = 4, color = 'k')
fig3.lines.extend([l1])

fig3.tight_layout()
fig3.show()

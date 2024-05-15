""" Illustrate the difficulty of visually analyzing changes in local resolution in noisy images, applied to SW-TPI Na MRI with different max gradient values and noise levels

    -load and display images already simulated and reconstructed using brainweb_grad_comparison_multi_patho.py

"""

import argparse
import json
from pathlib import Path
from copy import deepcopy
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndimage
import sigpy
import sys
from pymirc.image_operations import zoom3d

from utils import setup_blob_phantom, setup_brainweb_phantom, kb_rolloff, read_GE_ak_wav, tpi_kspace_coords_1_cm_scanner
from utils_sigpy import NUFFTT2starDualEchoModel, projected_gradient_operator

from scipy.ndimage import binary_erosion, zoom
import pymirc.viewer as pv
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    '--recon_type',
    type=str,
    default='bfgs',
    choices=[
        'bfgs_true_biexp', 'bfgs', 'adjoint', 'bfgs_highres_prior_true_biexp'
    ])
args = parser.parse_args()
recon_type = args.recon_type

# parameters
gradient_strengths = [16, 32, 48]
seeds = [191664964, 1662057957]
noise_levels = [0., 1.4e-2, 2.e-2]
recon_n = 128
no_decay = False

# read the data root directory from the config file
with open('.simulation_config.json', 'r') as f:
    data_root_dir: str = json.load(f)['data_root_dir']

# folder with simulated data
sim_dir = Path(data_root_dir) / (
    f'final_sim_brainweb_patho_glioma+lesion_gwm_cos' +
    ('_no_decay' if no_decay else ''))

save_dir = Path(
    '/uz/data/Admin/ngeresearch/MarinaFilipovic/Presentations/NaMRI/workInProgress/fig'
)

# load true image
im_true = np.abs(np.load(sim_dir / 'na_gt.npy'))
im_true = zoom(im_true, 0.5, order=1)

# loaded images
ims = np.zeros((len(gradient_strengths), len(seeds), len(noise_levels),
                recon_n, recon_n, recon_n))
# max intensity for display for each loaded image
max_vals = np.zeros((len(gradient_strengths), len(seeds), len(noise_levels)))

# load reconstructed images
for g, grad in enumerate(gradient_strengths):
    for s, seed in enumerate(seeds):
        for n, noise_level in enumerate(noise_levels):
            # no recons for this case, seed not relevant for noiseless data, so only one seed for 0. noise level
            if s > 0 and n == 0:
                continue
            # recon description
            if recon_type == 'adjoint':
                recon_params = ''
            # very low beta for noiseless data
            elif n == 0:
                recon_params = '_beta5.0E-05_it200'
            # mid beta
            else:
                recon_params = '_beta5.0E-03_it200'

            # folder with reconstructed images
            recon_dir = Path(data_root_dir) / (
                f'final_recon_brainweb_patho_glioma+lesion_gwm_cos_noise{noise_level:.1E}'
                + ('_no_decay' if no_decay else ''))
            # load magnitude image
            ims[g, s, n] = np.abs(
                np.load(
                    recon_dir /
                    f"recon_te1_{recon_type}{recon_params}_grad{grad}_seed{seed}.npz"
                )['recon'])
            # save max intensity
            max_vals[g, s, n] = np.max(ims[g, s, n])

# display
v = []

# images for display: mix of different gradients, different noise levels and different noise realizations
ims_show = [im_true, ims[0, 0, 1], ims[0, 0, 2], ims[2, 0, 1], ims[2, 1, 1]]

# empirical scaling for gridded recons based on max for noiseless gridded recons
if recon_type == 'adjoint':
    max_vals_show = [
        im_true.max(), max_vals[0, 0, 0].max(), max_vals[0, 0, 0].max(),
        max_vals[2, 0, 0].max(), max_vals[2, 0, 0].max()
    ]
    args = [{
        'cmap': 'viridis',
        'vmax': max_vals_show[i],
        'vmin': 0.
    } for i in range(len(ims_show))]
else:
    args = {'cmap': 'viridis', 'vmax': im_true.max(), 'vmin': 0.}

# display and save
v.append(
    pv.ThreeAxisViewer(
        ims_show,
        imshow_kwargs=args,
        ls='',
        rowlabels=['True', '', '', '', '']))
plt.savefig(
    save_dir / f'brainweb_{recon_type}_ambiguity_recons_grads.png',
    bbox_inches='tight')

# recons from noiseless data
ims_show = [im_true, ims[0, 0, 0], ims[1, 0, 0], ims[2, 0, 0]]
if recon_type == 'adjoint':
    max_vals_show = [
        im_true.max(), max_vals[0, 0, 0].max(), max_vals[1, 0, 0].max(),
        max_vals[2, 0, 0].max()
    ]
    args = [{
        'cmap': 'viridis',
        'vmax': max_vals_show[i],
        'vmin': 0.
    } for i in range(len(ims_show))]

v.append(
    pv.ThreeAxisViewer(
        ims_show, imshow_kwargs=args, ls='', rowlabels=['True', '', '', '']))
plt.savefig(
    save_dir / f'brainweb_{recon_type}_noiseless_recons_grads.png',
    bbox_inches='tight')

# noisy recons over different gradients
ims_show = [im_true, ims[0, 0, 1], ims[1, 0, 1], ims[2, 0, 1]]
if recon_type == 'adjoint':
    max_vals_show = [
        im_true.max(), max_vals[0, 0, 0].max(), max_vals[1, 0, 0].max(),
        max_vals[2, 0, 0].max()
    ]
    args = [{
        'cmap': 'viridis',
        'vmax': max_vals_show[i],
        'vmin': 0.
    } for i in range(len(ims_show))]

v.append(
    pv.ThreeAxisViewer(
        ims_show, imshow_kwargs=args, ls='', rowlabels=['True', '', '', '']))
plt.savefig(
    save_dir / f'brainweb_{recon_type}_recons_grads.png', bbox_inches='tight')

"""minimal script that shows how to solve L2square data fidelity + anatomical DTV prior"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from pymirc.image_operations import zoom3d
import pymirc.viewer as pv

parser = argparse.ArgumentParser()
parser.add_argument('--max_num_iter', type=int, default=200)
parser.add_argument('--noise_level', type=float, default=1e-2)
parser.add_argument('--phantom',
                    type=str,
                    default='brainweb',
                    choices=['brainweb', 'blob'])
parser.add_argument('--no_decay', action='store_true')
parser.add_argument('--no_decay_model', action='store_true')
args = parser.parse_args()

max_num_iter = args.max_num_iter
noise_level = args.noise_level
phantom = args.phantom
no_decay = args.no_decay
no_decay_model = args.no_decay_model

# read the data root directory from the config file
with open('.simulation_config.json', 'r') as f:
    data_root_dir: str = json.load(f)['data_root_dir']

odir = Path(data_root_dir) / 'gradient_test' / f'{phantom}_nodecay_{no_decay}_max_num_iter_{max_num_iter:04}'

betas = [0., 16, 32., 64., 128., 256., 512.]


#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

# load the ground truth image
gt = np.load(odir / 'na_gt.npy').real
roi_image = np.load(odir / 'roi_image.npy').real

zoomfac = gt.shape[0] / 128
# account for scaling factor due to data simulation and recon on different grids
scaling_fac = np.sqrt(zoomfac**3)

seeds = [x for x in range(1,76)] + [x for x in range(77,101)]

# arrays for noise-free recons
noise_free_gf_1 = np.zeros((len(betas),) + gt.shape)
noise_free_gf_3 = np.zeros((len(betas),) + gt.shape)

for ib, beta in enumerate(betas):
    noise_free_gf_1[ib,...] = zoom3d(np.abs(np.load(odir / f'recon_quad_prior_gf_01_nl_0.0E+00_beta_{beta:.1E}_s_001.npy')) / scaling_fac, zoomfac)
    noise_free_gf_3[ib,...] = zoom3d(np.abs(np.load(odir / f'recon_quad_prior_gf_03_nl_0.0E+00_beta_{beta:.1E}_s_001.npy')) / scaling_fac, zoomfac)

# calculate the bias of the noise_free recons

num_rois = roi_image.max() + 1

roi_dict = {0: 'background', 1:'CSF', 2:'GM', 3:'WM', 4:'eye1', 5:'eye2', 6:'glioma_center', 7:'glioma_ring', 8:'cosine'}

bias_noisefree_gf_1 = np.zeros((num_rois, len(betas)))
bias_noisefree_gf_3 = np.zeros((num_rois, len(betas)))

roi_inds = []
true_roi_means = np.zeros(num_rois)

for i in range(num_rois):
    roi_inds.append(np.where(roi_image == i))
    true_roi_means[i] = gt[roi_inds[i]].mean()

    for ib, beta in enumerate(betas):
        bias_noisefree_gf_1[i, ib] = noise_free_gf_1[ib,...][roi_inds[i]].mean() - true_roi_means[i]
        bias_noisefree_gf_3[i, ib] = noise_free_gf_3[ib,...][roi_inds[i]].mean() - true_roi_means[i]

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

# array for noise realizations
gf_1 = np.zeros((len(seeds), len(betas), 128, 128, 128), dtype = np.complex128)
gf_3 = np.zeros((len(seeds), len(betas), 128, 128, 128), dtype = np.complex128)

for iseed, seed in enumerate(seeds):
    print(f'loading seed {seed} {iseed+1} / {len(seeds)}', end = '\r')
    for ib, beta in enumerate(betas):
        gf_1[iseed, ib,...] = np.load(odir / f'recon_quad_prior_gf_01_nl_{noise_level:.1E}_beta_{beta:.1E}_s_{seed:03}.npy') / scaling_fac
        gf_3[iseed, ib,...] = np.load(odir / f'recon_quad_prior_gf_03_nl_{noise_level:.1E}_beta_{beta:.1E}_s_{seed:03}.npy') / scaling_fac

#gf_1_mean = np.abs(gf_1.mean(axis=0))
#gf_3_mean = np.abs(gf_3.mean(axis=0))
#gf_1_std = np.abs(gf_1.std(axis=0, ddof = 1))
#gf_3_std = np.abs(gf_3.std(axis=0, ddof = 1))

gf_1_mean = np.abs(gf_1).mean(axis=0)
gf_3_mean = np.abs(gf_3).mean(axis=0)
gf_1_std = np.abs(gf_1).std(axis=0, ddof = 1)
gf_3_std = np.abs(gf_3).std(axis=0, ddof = 1)

# interpolate the mean and std images to the ground truth grid

gf_1_mean_interp = np.zeros((len(betas),) + gt.shape)
gf_3_mean_interp = np.zeros((len(betas),) + gt.shape)
gf_1_std_interp = np.zeros((len(betas),) + gt.shape)
gf_3_std_interp = np.zeros((len(betas),) + gt.shape)

gf_1_first_interp = np.zeros((len(betas),) + gt.shape)
gf_3_first_interp = np.zeros((len(betas),) + gt.shape)

for i, _ in enumerate(betas):
    gf_1_mean_interp[i,...] = zoom3d(gf_1_mean[i,...], zoomfac)
    gf_3_mean_interp[i,...] = zoom3d(gf_3_mean[i,...], zoomfac)
    gf_1_std_interp[i,...] = zoom3d(gf_1_std[i,...], zoomfac)
    gf_3_std_interp[i,...] = zoom3d(gf_3_std[i,...], zoomfac)
    gf_1_first_interp[i,...] = zoom3d(np.abs(gf_1[0, i,...]), zoomfac)
    gf_3_first_interp[i,...] = zoom3d(np.abs(gf_3[0, i,...]), zoomfac)

# calculate bias and noise

bias_gf_1 = np.zeros((num_rois, len(betas)))
bias_gf_3 = np.zeros((num_rois, len(betas)))
noise_gf_1 = np.zeros((num_rois, len(betas)))
noise_gf_3 = np.zeros((num_rois, len(betas)))

for i in range(num_rois):
    for ib, beta in enumerate(betas):
        bias_gf_1[i, ib] = gf_1_mean_interp[ib,...][roi_inds[i]].mean() - true_roi_means[i]
        bias_gf_3[i, ib] = gf_3_mean_interp[ib,...][roi_inds[i]].mean() - true_roi_means[i]

        noise_gf_1[i, ib] = gf_1_std_interp[ib,...][roi_inds[i]].mean()
        noise_gf_3[i, ib] = gf_3_std_interp[ib,...][roi_inds[i]].mean()

#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------

# analyse cosine lesion slice by slice
start_sl = 189
stop_sl = 240

gt_cos_prof = np.zeros(stop_sl - start_sl)
gf_1_cos_prof = np.zeros((len(betas),stop_sl - start_sl))
gf_3_cos_prof = np.zeros((len(betas),stop_sl - start_sl))

for i, sl in enumerate(range(start_sl,stop_sl)):
    inds = np.where(roi_image[:,:,sl] == 8)
    gt_cos_prof[i] = gt[:,:,sl][inds].mean()

    for ib, beta in enumerate(betas):
        gf_1_cos_prof[ib,i] = noise_free_gf_1[ib,:,:,sl][inds].mean()
        gf_3_cos_prof[ib,i] = noise_free_gf_3[ib,:,:,sl][inds].mean()

fig3, ax3 = plt.subplots(1, len(betas), figsize=(14,2), sharey=True)

for ib, beta in enumerate(betas):
    ax3[ib].plot(gt_cos_prof, color = 'k')
    ax3[ib].plot(gf_1_cos_prof[ib,:], label = '0.16 G/cm')
    ax3[ib].plot(gf_3_cos_prof[ib,:], label = '0.48 G/cm')
    ax3[ib].grid(ls = ':')
    ax3[ib].set_title(f'beta = {int(beta)}', fontsize = 'medium')

ax3[-1].legend(loc = 3, fontsize = 'small', ncol=1)
fig3.tight_layout()
fig3.show()

snr_cos_1 = (gf_1_cos_prof.max(axis = 1) - gf_1_cos_prof.min(axis = 1)) / noise_gf_1[8,:]
snr_cos_3 = (gf_3_cos_prof.max(axis = 1) - gf_3_cos_prof.min(axis = 1)) / noise_gf_3[8,:]

#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------

# plot bias vs noise in different regions

fig, ax = plt.subplots(2,2, figsize = (6,6))

for i, iroi in enumerate([2,3,6,7]):
    ax.ravel()[i].plot(noise_gf_1[iroi,:], 100 * bias_noisefree_gf_1[iroi,:] / true_roi_means[iroi], '.-', label = '0.16 G/cm')
    ax.ravel()[i].plot(noise_gf_3[iroi,:], 100 * bias_noisefree_gf_3[iroi,:] / true_roi_means[iroi], '.-', label = '0.48 G/cm')
    ax.ravel()[i].set_title(roi_dict[iroi])
    ax.ravel()[i].grid(ls = ':')
    ax.ravel()[i].axhline(0, color = 'k')
    ax.ravel()[i].set_xlabel('ROI noise')
    ax.ravel()[i].set_ylabel('ROI bias [%]')

ax[0,0].legend(loc = 3)
fig.tight_layout()
fig.show()


fig2, ax2 = plt.subplots(2,2, figsize = (6,6))

for i, iroi in enumerate([2,3,6,7]):
    ax2.ravel()[i].plot(noise_gf_1[iroi,1:], 100 * bias_gf_1[iroi,1:] / true_roi_means[iroi], '.-', label = '0.16 G/cm')
    ax2.ravel()[i].plot(noise_gf_3[iroi,1:], 100 * bias_gf_3[iroi,1:] / true_roi_means[iroi], '.-', label = '0.48 G/cm')
    ax2.ravel()[i].set_title(roi_dict[iroi])
    ax2.ravel()[i].grid(ls = ':')
    ax2.ravel()[i].axhline(0, color = 'k')
    ax2.ravel()[i].set_xlabel('ROI noise')
    ax2.ravel()[i].set_ylabel('ROI bias [%]')

ax2[0,0].legend(loc = 3)
fig2.tight_layout()
fig2.show()




#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------


ims = dict(vmin = 0, vmax = 2, cmap = 'Greys_r')
vi = pv.ThreeAxisViewer([np.flip(x,2) for x in [np.tile(gt, (len(betas), 1, 1, 1)), np.abs(noise_free_gf_1), np.abs(noise_free_gf_3)]], 
                        imshow_kwargs=ims, width = 7, rowlabels = ['ground truth', '0.16 G/cm noise free', '0.48 G/cm noise free'])

vi2 = pv.ThreeAxisViewer([np.flip(x,2) for x in [np.abs(gf_1_mean), np.abs(gf_3_mean)]], 
                        imshow_kwargs=ims, width = 7, rowlabels =  ['0.16 G/cm mean', '0.48 G/cm noise mean'])

vi3 = pv.ThreeAxisViewer([np.flip(x,2) for x in [gf_1_first_interp, gf_3_first_interp]], 
                        imshow_kwargs=ims, width = 7, rowlabels =  ['0.16 G/cm 1st n.r.', '0.48 G/cm noise 1st n.r.'])

vi4 = pv.ThreeAxisViewer([np.flip(x,2) for x in [np.abs(gf_1_std), np.abs(gf_3_std)]], 
                        imshow_kwargs=dict(vmin=0,vmax=0.2, cmap = 'Greys_r'), 
                        width = 7, rowlabels =  ['0.16 G/cm std', '0.48 G/cm noise std'])

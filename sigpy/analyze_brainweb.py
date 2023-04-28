#TODO: - voxel-wise noise metric

import argparse
import json
from pathlib import Path
import numpy as np
import pymirc.viewer as pv
import nibabel as nib
from scipy.ndimage import binary_erosion, gaussian_filter, zoom
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--max_num_iter', type=int, default=1500)
parser.add_argument('--num_iter_r', type=int, default=100)
parser.add_argument('--noise_level', type=float, default=1e-2)
parser.add_argument('--phantom',
                    type=str,
                    default='brainweb',
                    choices=['brainweb', 'blob'])
parser.add_argument('--no_decay', action='store_true')
parser.add_argument('--regularization_norm_anatomical',
                    type=str,
                    default='L1',
                    choices=['L1', 'L2'])
parser.add_argument('--regularization_norm_non_anatomical',
                    type=str,
                    default='L2',
                    choices=['L1', 'L2'])
args = parser.parse_args()

max_num_iter = args.max_num_iter
num_iter_r = args.num_iter_r
noise_level = args.noise_level
phantom = args.phantom
no_decay = args.no_decay
regularization_norm_anatomical = args.regularization_norm_anatomical
regularization_norm_non_anatomical = args.regularization_norm_non_anatomical

beta_rs = [3e-2, 1e-1, 3e-1]
noise_metric = 2
#-----------------------------------------------------------------------

iter_shape = (128, 128, 128)
grid_shape = (64, 64, 64)

field_of_view_cm: float = 22.

# read the data root directory from the config file
with open('.simulation_config.json', 'r') as f:
    data_root_dir: str = json.load(f)['data_root_dir']

odirs = sorted(
    list((Path(data_root_dir) / 'run_brainweb').glob(
        f'{phantom}_nodecay_{no_decay}_i_{max_num_iter:04}_{num_iter_r:04}_nl_{noise_level:.1E}_s_*'
    )))

#odirs = odirs[:30]

#-----------------------------------------------------------------------
# load the ground truth
gt = np.abs(np.load(odirs[0] / 'na_gt.npy'))
sim_shape = gt.shape

# load the brainweb label array
phantom_data_path: Path = Path(data_root_dir) / 'brainweb54'

label_nii = nib.as_closest_canonical(
    nib.load(phantom_data_path / 'subject54_crisp_v.nii'))

# pad to 220mm FOV
lab_voxelsize = label_nii.header['pixdim'][1]
lab = np.asanyarray(label_nii.dataobj)
pad_size_220 = ((220 - np.array(lab.shape) * lab_voxelsize) / lab_voxelsize /
                2).astype(int)
pad_size_220 = ((pad_size_220[0], pad_size_220[0]),
                (pad_size_220[1], pad_size_220[1]), (pad_size_220[2],
                                                     pad_size_220[2]))
lab = np.pad(lab, pad_size_220, 'constant')

lab = zoom(lab, sim_shape[0] / lab.shape[0], order=0, prefilter=False)

# create a GM mask
gm_mask = (lab == 2).astype(np.uint8)
wm_mask = (lab == 3).astype(np.uint8)
csf_mask = (lab == 1).astype(np.uint8)

# load the aparc parcelation
aparc_nii = nib.as_closest_canonical(
    nib.load(phantom_data_path / 'aparc.DKTatlas+aseg_native.nii.gz'))
aparc = np.pad(np.asanyarray(aparc_nii.dataobj), pad_size_220, 'constant')

aparc = zoom(aparc, sim_shape[0] / aparc.shape[0], order=0, prefilter=False)

roi_inds = {}
roi_inds['cortical_gm'] = np.where((aparc * gm_mask) >= 1000)
roi_inds['eroded_cortical_gm'] = np.where(
    binary_erosion((aparc * gm_mask) >= 1000, iterations=1))
roi_inds['frontal'] = np.where(np.isin((aparc * gm_mask), [1028, 2028]))
roi_inds['eroded_frontal'] = np.where(
    binary_erosion(np.isin((aparc * gm_mask), [1028, 2028]), iterations=1))
roi_inds['temporal'] = np.where(
    np.isin((aparc * gm_mask), [1009, 1015, 1030, 1009, 1015, 1030]))
roi_inds['eroded_temporal'] = np.where(
    binary_erosion(np.isin((aparc * gm_mask),
                           [1009, 1015, 1030, 2009, 2015, 2030]),
                   iterations=1))
roi_inds['putamen'] = np.where(np.isin((aparc * gm_mask), [12, 51]))
roi_inds['eroded_putamen'] = np.where(
    binary_erosion(np.isin((aparc * gm_mask), [12, 51]), iterations=2))
roi_inds['ventricles'] = np.where(np.isin((aparc * csf_mask), [4, 43]))
roi_inds['eroded_ventricles'] = np.where(
    binary_erosion(np.isin((aparc * csf_mask), [4, 43]), iterations=2))
roi_inds['cerebellum'] = np.where(np.isin((aparc * csf_mask), [8, 15]))
roi_inds['eroded_cerebellum'] = np.where(
    binary_erosion(np.isin((aparc * csf_mask), [8, 15]), iterations=1))
roi_inds['wm'] = np.where(np.isin((aparc * wm_mask), [2, 41]))
roi_inds['eroded_wm'] = np.where(
    binary_erosion(np.isin((aparc * wm_mask), [2, 41]), iterations=3))
wm_border = np.zeros(sim_shape, dtype=np.uint8)
wm_border[np.isin((aparc * wm_mask), [2, 41])] = 1
wm_b1 = wm_border - binary_erosion(wm_border, iterations=1)
wm_b2 = wm_border - binary_erosion(wm_border, iterations=2)
roi_inds['wm_border1'] = np.where(wm_b1)
roi_inds['wm_border2'] = np.where(wm_b2)

betas_non_anatomical = [1e-2, 3e-2, 1e-1]
betas_anatomical = [3e-4, 1e-3, 3e-3]

sm_fwhms_mm = [0.1, 4., 6., 8., 10.]

recons_e1_no_decay = np.zeros((
    len(odirs),
    len(betas_non_anatomical),
) + sim_shape)
agrs_e1_no_decay = np.zeros((
    len(odirs),
    len(betas_anatomical),
) + sim_shape)

agrs_both_echos_w_decay0 = np.zeros((
    len(odirs),
    len(betas_anatomical),
) + sim_shape)

agrs_both_echos_w_decay1 = np.zeros((
    len(odirs),
    len(betas_anatomical),
) + sim_shape)

agrs_both_echos_w_decay2 = np.zeros((
    len(odirs),
    len(betas_anatomical),
) + sim_shape)

r0s = np.zeros((
    len(odirs),
    len(betas_anatomical),
) + sim_shape)

r1s = np.zeros((
    len(odirs),
    len(betas_anatomical),
) + sim_shape)

r2s = np.zeros((
    len(odirs),
    len(betas_anatomical),
) + sim_shape)

#ifft1s = np.zeros((
#    len(odirs),
#    len(sm_fwhms_mm),
#) + sim_shape)
#
#ifft_scale_fac = 1.

# calculate the ROI averages
true_means = {}
recon_e1_no_decay_roi_means = {}
agr_e1_no_decay_roi_means = {}
agr_both_echos_w_decay0_roi_means = {}
agr_both_echos_w_decay1_roi_means = {}
agr_both_echos_w_decay2_roi_means = {}

recon_e1_no_decay_roi_stds = {}
agr_e1_no_decay_roi_stds = {}
agr_both_echos_w_decay0_roi_stds = {}
agr_both_echos_w_decay1_roi_stds = {}
agr_both_echos_w_decay2_roi_stds = {}

ifft1_roi_means = {}
sl = 73

# calculate the true means
for key, inds in roi_inds.items():
    true_means[key] = gt[inds].mean()
#    ifft1_roi_means[key] = np.array([[x[inds].mean() for x in y]
#                                     for y in ifft1s])

#for i, odir in enumerate(odirs):
#    print('loading IFFTs', odir)
#    # load IFFT of first echo
#    ifft = ifft_scale_fac * np.load(odir / 'ifft1.npy')
#    ifft_voxsize = 220 / ifft.shape[0]
#
#    for j, sm_fwhm_mm in enumerate(sm_fwhms_mm):
#        ifft1s[i, j, ...] = zoom(np.abs(
#            gaussian_filter(ifft, sm_fwhm_mm / (2.35 * ifft_voxsize))),
#                                 sim_shape[0] / ifft.shape[0],
#                                 order=1,
#                                 prefilter=False)
#
#ifft1s_mean = ifft1s.mean(axis=0)
#ifft1s_std = ifft1s.std(axis=0)

# load the image scale factor
with open(odirs[0] / 'scaling_factors.json', 'r') as f:
    image_scale = json.load(f)['image_scale']

for i, odir in enumerate(odirs):
    print('loading iterative no decay model', odir)

    # load iterative recons of first echo with non-anatomical prior
    for ib, beta_non_anatomical in enumerate(betas_non_anatomical):
        ofile_e1_no_decay = odir / f'recon_echo_1_no_decay_model_{regularization_norm_non_anatomical}_{beta_non_anatomical:.1E}_{max_num_iter}.npz'
        d = np.load(ofile_e1_no_decay)
        recons_e1_no_decay[i, ib, ...] = zoom(np.abs(d['x'] / image_scale),
                                              sim_shape[0] / iter_shape[0],
                                              order=1,
                                              prefilter=False)

for key, inds in roi_inds.items():
    recon_e1_no_decay_roi_means[key] = np.array([[x[inds].mean() for x in y]
                                                 for y in recons_e1_no_decay])

recons_e1_no_decay_mean = recons_e1_no_decay.mean(axis=0)
recons_e1_no_decay_std = recons_e1_no_decay.std(axis=0)

for key, inds in roi_inds.items():
    recon_e1_no_decay_roi_stds[key] = np.array(
        [x[inds].mean() for x in recons_e1_no_decay_std])

for i, odir in enumerate(odirs):
    print('loading AGR no decay model', odir)
    # load AGR of first echo with out decay model
    for ib, beta_anatomical in enumerate(betas_anatomical):
        ofile_e1_no_decay_agr = odir / f'agr_echo_1_no_decay_model_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{max_num_iter}.npz'
        d = np.load(ofile_e1_no_decay_agr)
        agrs_e1_no_decay[i, ib, ...] = zoom(np.abs(d['x'] / image_scale),
                                            sim_shape[0] / iter_shape[0],
                                            order=1,
                                            prefilter=False)

for key, inds in roi_inds.items():
    agr_e1_no_decay_roi_means[key] = np.array([[x[inds].mean() for x in y]
                                               for y in agrs_e1_no_decay])

agrs_e1_no_decay_mean = agrs_e1_no_decay.mean(axis=0)
agrs_e1_no_decay_std = agrs_e1_no_decay.std(axis=0)

for i, odir in enumerate(odirs):
    print('loading AGR w decay model', odir)
    # load AGR of borhs echos with decay model
    for ib, beta_anatomical in enumerate(betas_anatomical):
        ofile_both_echos_agr0 = odir / f'agr_both_echo_w_decay_model_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{beta_rs[0]:.1E}_{max_num_iter}_{num_iter_r}.npz'
        ofile_both_echos_agr1 = odir / f'agr_both_echo_w_decay_model_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{beta_rs[1]:.1E}_{max_num_iter}_{num_iter_r}.npz'
        ofile_both_echos_agr2 = odir / f'agr_both_echo_w_decay_model_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{beta_rs[2]:.1E}_{max_num_iter}_{num_iter_r}.npz'

        outfile_r0 = odir / f'est_ratio_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{beta_rs[0]:.1E}_{max_num_iter}_{num_iter_r}.npy'
        outfile_r1 = odir / f'est_ratio_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{beta_rs[1]:.1E}_{max_num_iter}_{num_iter_r}.npy'
        outfile_r2 = odir / f'est_ratio_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{beta_rs[2]:.1E}_{max_num_iter}_{num_iter_r}.npy'

        d0 = np.load(ofile_both_echos_agr0)
        d1 = np.load(ofile_both_echos_agr1)
        d2 = np.load(ofile_both_echos_agr2)

        agrs_both_echos_w_decay0[i, ib,
                                 ...] = zoom(np.abs(d0['x'] / image_scale),
                                             sim_shape[0] / iter_shape[0],
                                             order=1,
                                             prefilter=False)
        agrs_both_echos_w_decay1[i, ib,
                                 ...] = zoom(np.abs(d1['x'] / image_scale),
                                             sim_shape[0] / iter_shape[0],
                                             order=1,
                                             prefilter=False)
        agrs_both_echos_w_decay2[i, ib,
                                 ...] = zoom(np.abs(d2['x'] / image_scale),
                                             sim_shape[0] / iter_shape[0],
                                             order=1,
                                             prefilter=False)

        r0s[i, ib, ...] = zoom(np.load(outfile_r0),
                               sim_shape[0] / iter_shape[0],
                               order=1,
                               prefilter=False)

        r1s[i, ib, ...] = zoom(np.load(outfile_r1),
                               sim_shape[0] / iter_shape[0],
                               order=1,
                               prefilter=False)

        r2s[i, ib, ...] = zoom(np.load(outfile_r2),
                               sim_shape[0] / iter_shape[0],
                               order=1,
                               prefilter=False)

for key, inds in roi_inds.items():
    agr_both_echos_w_decay0_roi_means[key] = np.array(
        [[x[inds].mean() for x in y] for y in agrs_both_echos_w_decay0])
    agr_both_echos_w_decay1_roi_means[key] = np.array(
        [[x[inds].mean() for x in y] for y in agrs_both_echos_w_decay1])
    agr_both_echos_w_decay2_roi_means[key] = np.array(
        [[x[inds].mean() for x in y] for y in agrs_both_echos_w_decay2])

agrs_both_echos_w_decay0_mean = agrs_both_echos_w_decay0.mean(axis=0)
agrs_both_echos_w_decay1_mean = agrs_both_echos_w_decay1.mean(axis=0)
agrs_both_echos_w_decay2_mean = agrs_both_echos_w_decay2.mean(axis=0)

agrs_both_echos_w_decay0_std = agrs_both_echos_w_decay0.std(axis=0)
agrs_both_echos_w_decay1_std = agrs_both_echos_w_decay1.std(axis=0)
agrs_both_echos_w_decay2_std = agrs_both_echos_w_decay2.std(axis=0)

#--------------------------------------------------------------------------------
# bias noise plots

num_rows = 2
num_cols = len(roi_inds) // 2
fig, ax = plt.subplots(num_rows,
                       num_cols,
                       figsize=(4 * num_cols, 4 * num_rows),
                       sharey=True)

#for i, (roi, vals) in enumerate(ifft1_roi_means.items()):
#    x = vals.std(0)
#    y = 100 * (vals.mean(0) - true_means[roi]) / true_means[roi]
#    ax.ravel()[i].plot(x, y, 'x-', label='IFFT')
#
#    for j in range(len(x)):
#        ax.ravel()[i].annotate(f'{sm_fwhms_mm[j]:.1f}mm', (x[j], y[j]),
#                               horizontalalignment='center',
#                               verticalalignment='bottom')

for i, (roi, vals) in enumerate(recon_e1_no_decay_roi_means.items()):
    if noise_metric == 1:
        x = vals.std(0)
    else:
        x = np.array([z[roi_inds[roi]].mean() for z in recons_e1_no_decay_std])

    y = 100 * (vals.mean(0) - true_means[roi]) / true_means[roi]
    ax.ravel()[i].plot(x, y, 'o-', label='iter quad. prior')

    for j in range(len(x)):
        ax.ravel()[i].annotate(f'{betas_non_anatomical[j]:.1E}', (x[j], y[j]),
                               horizontalalignment='center',
                               verticalalignment='bottom',
                               fontsize='x-small')

for i, (roi, vals) in enumerate(agr_e1_no_decay_roi_means.items()):
    if noise_metric == 1:
        x = vals.std(0)
    else:
        x = np.array([z[roi_inds[roi]].mean() for z in agrs_e1_no_decay_std])

    y = 100 * (vals.mean(0) - true_means[roi]) / true_means[roi]
    ax.ravel()[i].plot(x, y, 'o-', label='AGR wo decay m.')

    for j in range(len(x)):
        ax.ravel()[i].annotate(f'{betas_anatomical[j]:.1E}', (x[j], y[j]),
                               horizontalalignment='center',
                               verticalalignment='bottom',
                               fontsize='x-small')

for i, (roi, vals) in enumerate(agr_both_echos_w_decay0_roi_means.items()):
    if noise_metric == 1:
        x = vals.std(0)
    else:
        x = np.array(
            [z[roi_inds[roi]].mean() for z in agrs_both_echos_w_decay0_std])
    y = 100 * (vals.mean(0) - true_means[roi]) / true_means[roi]
    ax.ravel()[i].plot(x, y, 'o-', label=f'AGR w decay m. {beta_rs[0]:.1E}')

    for j in range(len(x)):
        ax.ravel()[i].annotate(f'{betas_anatomical[j]:.1E}', (x[j], y[j]),
                               horizontalalignment='center',
                               verticalalignment='bottom',
                               fontsize='x-small')

for i, (roi, vals) in enumerate(agr_both_echos_w_decay1_roi_means.items()):
    if noise_metric == 1:
        x = vals.std(0)
    else:
        x = np.array(
            [z[roi_inds[roi]].mean() for z in agrs_both_echos_w_decay1_std])
    y = 100 * (vals.mean(0) - true_means[roi]) / true_means[roi]
    ax.ravel()[i].plot(x, y, 'o-', label=f'AGR w decay m. {beta_rs[1]:.1E}')

for i, (roi, vals) in enumerate(agr_both_echos_w_decay2_roi_means.items()):
    if noise_metric == 1:
        x = vals.std(0)
    else:
        x = np.array(
            [z[roi_inds[roi]].mean() for z in agrs_both_echos_w_decay2_std])
    y = 100 * (vals.mean(0) - true_means[roi]) / true_means[roi]
    ax.ravel()[i].plot(x, y, 'o-', label=f'AGR w decay m. {beta_rs[2]:.1E}')

    ax.ravel()[i].set_title(roi)
    ax.ravel()[i].grid(ls=':')
    if noise_metric == 1:
        ax.ravel()[i].set_xlabel('std.dev. of ROI mean')
    else:
        ax.ravel()[i].set_xlabel('ROI averaged std.dev.')
    ax.ravel()[i].axhline(0, color='k')

    ## get y-axis limits of the plot
    #low, high = ax.ravel()[i].get_ylim()
    ## find the new limits
    #bound = max(abs(low), abs(high))
    ## set new limits
    #ax.ravel()[i].set_ylim(-bound, bound)

ax[0, 0].set_ylabel('bias of ROI mean [%]')
ax[1, 0].set_ylabel('bias of ROI mean [%]')
ax.ravel()[-1].legend(loc='lower right', fontsize='small')

fig.tight_layout()
fig.show()

#--------------------------------------------------------------------------------
# recon plots

num_cols2 = len(betas_non_anatomical)
num_rows2 = 3
fig2, ax2 = plt.subplots(num_rows2,
                         num_cols2,
                         figsize=(3 * num_cols2, 3 * num_rows2))

for i in range(num_cols2):
    ax2[0, i].imshow(recons_e1_no_decay[0, i, ..., sl].T,
                     origin='lower',
                     cmap='Greys_r',
                     vmin=0,
                     vmax=2)
    ax2[1, i].imshow(recons_e1_no_decay_mean[i, ..., sl].T - gt[:, :, sl].T,
                     origin='lower',
                     cmap='seismic',
                     vmin=-1,
                     vmax=1)
    ax2[2, i].imshow(recons_e1_no_decay_std[i, ..., sl].T,
                     origin='lower',
                     cmap='Greys_r',
                     vmin=0,
                     vmax=0.5)

    ax2[0, i].set_title(f'beta = {betas_non_anatomical[i]}')

ax2[0, 0].set_ylabel('first noise realization')
ax2[1, 0].set_ylabel('bias image')
ax2[2, 0].set_ylabel('std.dev. image')

for axx in ax2.ravel():
    axx.set_xticks([])
    axx.set_yticks([])

fig2.tight_layout()
fig2.show()

#------------------------------------------------------------------------------------

#num_cols3 = len(sm_fwhms_mm)
#num_rows3 = 3
#fig3, ax3 = plt.subplots(num_rows3,
#                         num_cols3,
#                         figsize=(3 * num_cols3, 3 * num_rows3))
#
#for i in range(num_cols3):
#    ax3[0, i].imshow(ifft1s[0,i, ..., sl].T,
#                     origin='lower',
#                     cmap='Greys_r',
#                     vmin=0,
#                     vmax=2)
#    ax3[1, i].imshow(ifft1s_mean[i, ..., sl].T - gt[:, :, sl].T,
#                     origin='lower',
#                     cmap='seismic',
#                     vmin=-1,
#                     vmax=1)
#    ax3[2, i].imshow(ifft1s_std[i, ..., sl].T,
#                     origin='lower',
#                     cmap='Greys_r',
#                     vmin=0,
#                     vmax=0.5)
#
#    ax3[0, i].set_title(f'fwhm = {sm_fwhms_mm[i]:.1f}mm')
#
#ax3[0, 0].set_ylabel('first noise realization')
#ax3[1, 0].set_ylabel('bias image')
#ax3[2, 0].set_ylabel('std.dev. image')
#
#for axx in ax3.ravel():
#    axx.set_xticks([])
#    axx.set_yticks([])
#
#fig3.tight_layout()
#fig3.show()

#--------------------------------------------------------------------------------

num_cols4 = len(betas_anatomical)
num_rows4 = 3
fig4, ax4 = plt.subplots(num_rows4,
                         num_cols4,
                         figsize=(3 * num_cols4, 3 * num_rows4))

for i in range(num_cols4):
    ax4[0, i].imshow(agrs_e1_no_decay[0, i, ..., sl].T,
                     origin='lower',
                     cmap='Greys_r',
                     vmin=0,
                     vmax=2)
    ax4[1, i].imshow(agrs_e1_no_decay_mean[i, ..., sl].T - gt[:, :, sl].T,
                     origin='lower',
                     cmap='seismic',
                     vmin=-1,
                     vmax=1)
    ax4[2, i].imshow(agrs_e1_no_decay_std[i, ..., sl].T,
                     origin='lower',
                     cmap='Greys_r',
                     vmin=0,
                     vmax=0.5)

    ax4[0, i].set_title(f'beta = {betas_anatomical[i]}')

ax4[0, 0].set_ylabel('first noise realization')
ax4[1, 0].set_ylabel('bias image')
ax4[2, 0].set_ylabel('std.dev. image')

for axx in ax4.ravel():
    axx.set_xticks([])
    axx.set_yticks([])

fig4.tight_layout()
fig4.show()

#--------------------------------------------------------------------------------

num_cols5 = len(betas_anatomical)
num_rows5 = 3
fig5, ax5 = plt.subplots(num_rows5,
                         num_cols5,
                         figsize=(3 * num_cols5, 3 * num_rows5))

for i in range(num_cols5):
    ax5[0, i].imshow(agrs_both_echos_w_decay0[0, i, ..., sl].T,
                     origin='lower',
                     cmap='Greys_r',
                     vmin=0,
                     vmax=2)
    ax5[1,
        i].imshow(agrs_both_echos_w_decay0_mean[i, ..., sl].T - gt[:, :, sl].T,
                  origin='lower',
                  cmap='seismic',
                  vmin=-1,
                  vmax=1)
    ax5[2, i].imshow(agrs_both_echos_w_decay0_std[i, ..., sl].T,
                     origin='lower',
                     cmap='Greys_r',
                     vmin=0,
                     vmax=0.5)

    ax5[0, i].set_title(f'beta = {betas_anatomical[i]}')

ax5[0, 0].set_ylabel('first noise realization')
ax5[1, 0].set_ylabel('bias image')
ax5[2, 0].set_ylabel('std.dev. image')

for axx in ax5.ravel():
    axx.set_xticks([])
    axx.set_yticks([])

fig5.tight_layout()
fig5.show()

#--------------------------------------------------------------------------------

fig6, ax6 = plt.subplots(num_rows5,
                         num_cols5,
                         figsize=(3 * num_cols5, 3 * num_rows5))

for i in range(num_cols5):
    ax6[0, i].imshow(agrs_both_echos_w_decay1[0, i, ..., sl].T,
                     origin='lower',
                     cmap='Greys_r',
                     vmin=0,
                     vmax=2)
    ax6[1,
        i].imshow(agrs_both_echos_w_decay1_mean[i, ..., sl].T - gt[:, :, sl].T,
                  origin='lower',
                  cmap='seismic',
                  vmin=-1,
                  vmax=1)
    ax6[2, i].imshow(agrs_both_echos_w_decay1_std[i, ..., sl].T,
                     origin='lower',
                     cmap='Greys_r',
                     vmin=0,
                     vmax=0.5)

    ax6[0, i].set_title(f'beta = {betas_anatomical[i]}')

ax6[0, 0].set_ylabel('first noise realization')
ax6[1, 0].set_ylabel('bias image')
ax6[2, 0].set_ylabel('std.dev. image')

for axx in ax6.ravel():
    axx.set_xticks([])
    axx.set_yticks([])

fig6.tight_layout()
fig6.show()

#--------------------------------------------------------------------------------

fig7, ax7 = plt.subplots(num_rows5,
                         num_cols5,
                         figsize=(3 * num_cols5, 3 * num_rows5))

for i in range(num_cols5):
    ax7[0, i].imshow(agrs_both_echos_w_decay2[0, i, ..., sl].T,
                     origin='lower',
                     cmap='Greys_r',
                     vmin=0,
                     vmax=2)
    ax7[1,
        i].imshow(agrs_both_echos_w_decay2_mean[i, ..., sl].T - gt[:, :, sl].T,
                  origin='lower',
                  cmap='seismic',
                  vmin=-1,
                  vmax=1)
    ax7[2, i].imshow(agrs_both_echos_w_decay2_std[i, ..., sl].T,
                     origin='lower',
                     cmap='Greys_r',
                     vmin=0,
                     vmax=0.5)

    ax7[0, i].set_title(f'beta = {betas_anatomical[i]}')

ax7[0, 0].set_ylabel('first noise realization')
ax7[1, 0].set_ylabel('bias image')
ax7[2, 0].set_ylabel('std.dev. image')

for axx in ax7.ravel():
    axx.set_xticks([])
    axx.set_yticks([])

fig7.tight_layout()
fig7.show()

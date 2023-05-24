#TODO: - voxel-wise noise metric

import argparse
import json
from pathlib import Path
import numpy as np
import pymirc.viewer as pv
import nibabel as nib
from scipy.ndimage import binary_erosion, gaussian_filter, zoom, center_of_mass
from collections import OrderedDict
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_context('paper')

parser = argparse.ArgumentParser()
parser.add_argument('--max_num_iter', type=int, default=2000)
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

beta_rs = [1e-1, 3e-1, 3e-1]
noise_metric = 2

betas_non_anatomical = [1e-2, 3e-2, 1e-1]
betas_anatomical = [3e-4, 1e-3, 3e-3]
sm_fwhms_mm = [0.1, 4., 6., 8., 10.]
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

odirs = odirs[:18]

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

roi_inds = OrderedDict()
roi_inds['ventricles'] = np.where(np.isin((aparc * csf_mask), [4, 43]))
roi_inds['white matter'] = np.where(np.isin((aparc * wm_mask), [2, 41]))
roi_inds['putamen'] = np.where(np.isin((aparc * gm_mask), [12, 51]))
roi_inds['cerebellum'] = np.where(np.isin((aparc * gm_mask), [8, 47]))
roi_inds['cortical grey matter'] = np.where((aparc * gm_mask) >= 1000)
roi_inds['frontal'] = np.where(np.isin((aparc * gm_mask), [1028, 2028]))
roi_inds['temporal'] = np.where(
    np.isin((aparc * gm_mask), [1009, 1015, 1030, 2009, 2015, 2030]))

# visualize the ROIs
vis = []

for key, inds in roi_inds.items():
    roi_mask = np.zeros_like(gt)
    roi_mask[inds] = 1

    com = [int(x) for x in center_of_mass(roi_mask)]
    sl2 = np.argmax(roi_mask.sum((0, 1)))
    sl1 = np.argmax(roi_mask.sum((0, 2)))

    fig0, ax0 = plt.subplots(1, 2, figsize=(6, 3))
    ax0[0].imshow(gt[:, :, sl2].T, origin='lower', cmap='Greys_r')
    ax0[0].contour(roi_mask[:, :, sl2].T, origin='lower', levels=1, cmap='hot')
    ax0[1].imshow(gt[:, sl1, :].T, origin='lower', cmap='Greys_r')
    ax0[1].contour(roi_mask[:, sl1, :].T, origin='lower', levels=1, cmap='hot')
    ax0[0].set_axis_off()
    ax0[1].set_axis_off()
    fig0.tight_layout()
    fig0.savefig(odirs[0] / f'roi_{key}.png', dpi=300, bbox_inches='tight')
    plt.close(fig0)

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

# calculate the ROI averages
true_means = {}
recon_e1_no_decay_roi_means = {}
agr_e1_no_decay_roi_means = {}
agr_both_echos_w_decay0_roi_means = {}
agr_both_echos_w_decay1_roi_means = {}

recon_e1_no_decay_roi_stds = {}
agr_e1_no_decay_roi_stds = {}
agr_both_echos_w_decay0_roi_stds = {}
agr_both_echos_w_decay1_roi_stds = {}

ifft1_roi_means = {}
sl = int(0.4375 * gt.shape[0])

# calculate the true means
for key, inds in roi_inds.items():
    true_means[key] = gt[inds].mean()

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

agrs_both_echos_w_decay0_mean = agrs_both_echos_w_decay0.mean(axis=0)
agrs_both_echos_w_decay1_mean = agrs_both_echos_w_decay1.mean(axis=0)

agrs_both_echos_w_decay0_std = agrs_both_echos_w_decay0.std(axis=0)
agrs_both_echos_w_decay1_std = agrs_both_echos_w_decay1.std(axis=0)

# calculate the RMSE in all ROIs
rmse_recon_e1_no_decay = {}
rmse_agr_e1_no_decay = {}
rmse_agr_both_echos_w_decay0 = {}
rmse_agr_both_echos_w_decay1 = {}

for key, inds in roi_inds.items():
    rmse_recon_e1_no_decay[key] = np.array(
        [[np.sqrt(np.mean((x[inds] - gt[inds])**2)) for x in y]
         for y in recons_e1_no_decay])
    rmse_agr_e1_no_decay[key] = np.array(
        [[np.sqrt(np.mean((x[inds] - gt[inds])**2)) for x in y]
         for y in agrs_e1_no_decay])
    rmse_agr_both_echos_w_decay0[key] = np.array(
        [[np.sqrt(np.mean((x[inds] - gt[inds])**2)) for x in y]
         for y in agrs_both_echos_w_decay0])
    rmse_agr_both_echos_w_decay1[key] = np.array(
        [[np.sqrt(np.mean((x[inds] - gt[inds])**2)) for x in y]
         for y in agrs_both_echos_w_decay1])

#--------------------------------------------------------------------------------
# bias noise plots

num_rows = 1
num_cols = len(roi_inds)
fig, ax = plt.subplots(num_rows,
                       num_cols,
                       figsize=(2.5 * num_cols, 3.5 * num_rows),
                       sharex=True,
                       sharey=True)

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

    ax.ravel()[i].set_title(roi)
    ax.ravel()[i].grid(ls=':')
    if noise_metric == 1:
        ax.ravel()[i].set_xlabel('std.dev. of ROI mean')
    else:
        ax.ravel()[i].set_xlabel('ROI averaged std.dev.')
    ax.ravel()[i].axhline(0, color='k')

ax[0].set_ylabel('bias of ROI mean [%]')
ax[0].set_ylabel('bias of ROI mean [%]')
ax.ravel()[-1].legend(loc='lower right', fontsize='small')

fig.tight_layout()
fig.show()

#--------------------------------------------------------------------------------
# RMSE plots
#------------------------------------------------------------------------------------

box_kwargs = dict(showfliers=False,
                  showbox=False,
                  showcaps=False,
                  showmeans=True,
                  meanline=True,
                  meanprops=dict(color='k', ls='-'),
                  medianprops=dict(visible=False),
                  whiskerprops=dict(visible=False))

swarm_kwargs = dict(palette='dark:gray', size=1.5)

num_rows = 4
num_cols = len(roi_inds)
fig2, ax2 = plt.subplots(num_rows,
                         num_cols,
                         figsize=(2.5 * num_cols, 2.5 * num_rows),
                         sharex=True,
                         sharey='col')

for i, roi in enumerate(roi_inds.keys()):
    print(i, roi)
    sns.boxplot(rmse_recon_e1_no_decay[roi], ax=ax2[0, i], **box_kwargs)
    sns.swarmplot(rmse_recon_e1_no_decay[roi], ax=ax2[0, i], **swarm_kwargs)
    sns.boxplot(rmse_agr_e1_no_decay[roi], ax=ax2[1, i], **box_kwargs)
    sns.swarmplot(rmse_agr_e1_no_decay[roi], ax=ax2[1, i], **swarm_kwargs)
    sns.boxplot(rmse_agr_both_echos_w_decay0[roi], ax=ax2[2, i], **box_kwargs)
    sns.swarmplot(rmse_agr_both_echos_w_decay0[roi],
                  ax=ax2[2, i],
                  **swarm_kwargs)
    sns.boxplot(rmse_agr_both_echos_w_decay1[roi], ax=ax2[3, i], **box_kwargs)
    sns.swarmplot(rmse_agr_both_echos_w_decay1[roi],
                  ax=ax2[3, i],
                  **swarm_kwargs)
    ax2[0, i].set_title(roi)

ax2[0, 0].set_ylabel('RMSE iter. quad prior')
ax2[1, 0].set_ylabel('RMSE AGR wo decay m.')
ax2[2, 0].set_ylabel(f'RMSE AGR w decay m. {beta_rs[0]:.1E}')
ax2[3, 0].set_ylabel(f'RMSE AGR w decay m. {beta_rs[1]:.1E}')

for axx in ax2.ravel():
    axx.grid(ls=':')

fig2.tight_layout()
fig2.show()
#--------------------------------------------------------------------------------
# recon plots
#------------------------------------------------------------------------------------

vmax_std = 0.2
vmax = 1.75
bmax = 1
inoise = 0

fig3a, ax3a = plt.subplots(3, 3, figsize=(6, 6))
fig3b, ax3b = plt.subplots(3, 3, figsize=(6, 6))
fig3c, ax3c = plt.subplots(3, 3, figsize=(6, 6))

for i in range(3):
    i0 = ax3a[i, 0].imshow(recons_e1_no_decay[inoise, i, ..., sl].T,
                           origin='lower',
                           cmap='Greys_r',
                           vmin=0,
                           vmax=vmax)
    i1 = ax3a[i,
              1].imshow(recons_e1_no_decay_mean[i, ..., sl].T - gt[:, :, sl].T,
                        origin='lower',
                        cmap='seismic',
                        vmin=-bmax,
                        vmax=bmax)
    i2 = ax3a[i, 2].imshow(recons_e1_no_decay_std[i, ..., sl].T,
                           origin='lower',
                           cmap='Greys_r',
                           vmin=0,
                           vmax=vmax_std)
    i3 = ax3b[i, 0].imshow(agrs_e1_no_decay[inoise, i, ..., sl].T,
                           origin='lower',
                           cmap='Greys_r',
                           vmin=0,
                           vmax=vmax)
    i4 = ax3b[i,
              1].imshow(agrs_e1_no_decay_mean[i, ..., sl].T - gt[:, :, sl].T,
                        origin='lower',
                        cmap='seismic',
                        vmin=-bmax,
                        vmax=bmax)
    i5 = ax3b[i, 2].imshow(agrs_e1_no_decay_std[i, ..., sl].T,
                           origin='lower',
                           cmap='Greys_r',
                           vmin=0,
                           vmax=vmax_std)
    i6 = ax3c[i, 0].imshow(agrs_both_echos_w_decay1[inoise, i, ..., sl].T,
                           origin='lower',
                           cmap='Greys_r',
                           vmin=0,
                           vmax=vmax)
    i7 = ax3c[i, 1].imshow(agrs_both_echos_w_decay1_mean[i, ..., sl].T -
                           gt[:, :, sl].T,
                           origin='lower',
                           cmap='seismic',
                           vmin=-bmax,
                           vmax=bmax)
    i8 = ax3c[i, 2].imshow(agrs_both_echos_w_decay1_std[i, ..., sl].T,
                           origin='lower',
                           cmap='Greys_r',
                           vmin=0,
                           vmax=vmax_std)

ax3a[0, 0].set_title('first noise realization')
ax3a[0, 1].set_title('bias image')
ax3a[0, 2].set_title('std.dev. image')
ax3b[0, 0].set_title('first noise realization')
ax3b[0, 1].set_title('bias image')
ax3b[0, 2].set_title('std.dev. image')
ax3c[0, 0].set_title('first noise realization')
ax3c[0, 1].set_title('bias image')
ax3c[0, 2].set_title('std.dev. image')

for axx in ax3a.ravel():
    axx.set_axis_off()
for axx in ax3b.ravel():
    axx.set_axis_off()
for axx in ax3c.ravel():
    axx.set_axis_off()

fig3a.tight_layout()
fig3b.tight_layout()
fig3c.tight_layout()

fig3a.savefig('conv_sim.png')
fig3b.savefig('agr_wo_decay_sim.png')
fig3c.savefig('agr_w_decay_sim.png')

fig3a.show()
fig3b.show()
fig3c.show()

fig4, ax4 = plt.subplots(1, 1, figsize=(2, 2))
ax4.imshow(gt[..., sl].T, origin='lower', cmap='Greys_r', vmin=0, vmax=vmax)
ax4.set_axis_off()
fig4.tight_layout()
fig4.show()

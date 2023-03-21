import argparse
import json
from pathlib import Path
import numpy as np
import pymirc.viewer as pv
import nibabel as nib
from pymirc.image_operations import zoom3d
from scipy.ndimage import binary_erosion, gaussian_filter
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--max_num_iter', type=int, default=1000)
parser.add_argument('--noise_level', type=float, default=2e-2)
parser.add_argument('--phantom',
                    type=str,
                    default='brainweb',
                    choices=['brainweb', 'blob'])
parser.add_argument('--no_decay', action='store_true')
args = parser.parse_args()

max_num_iter = args.max_num_iter
noise_level = args.noise_level
phantom = args.phantom
no_decay = args.no_decay

#-----------------------------------------------------------------------

sim_shape = (160, 160, 160)
iter_shape = (128, 128, 128)
grid_shape = (64, 64, 64)

field_of_view_cm: float = 22.

odirs = sorted(
    list(
        Path('run_brainweb').glob(
            f'{phantom}_nodecay_{no_decay}_i_{max_num_iter:04}_nl_{noise_level:.1E}_s_*'
        )))

#-----------------------------------------------------------------------
# load the brainweb label array

with open('.simulation_config.json', 'r') as f:
    data_root_dir: str = json.load(f)['data_root_dir']

phantom_data_path: Path = Path(data_root_dir) / 'brainweb54'

label_nii = nib.as_closest_canonical(
    nib.load(phantom_data_path / 'subject54_crisp_v.nii'))

# pad to 220mm FOV
lab_voxelsize = label_nii.header['pixdim'][1]
lab = label_nii.get_fdata()
pad_size_220 = ((220 - np.array(lab.shape) * lab_voxelsize) / lab_voxelsize /
                2).astype(int)
pad_size_220 = ((pad_size_220[0], pad_size_220[0]),
                (pad_size_220[1], pad_size_220[1]), (pad_size_220[2],
                                                     pad_size_220[2]))
lab = np.pad(lab, pad_size_220, 'constant')

# load the aparc parcelation
aparc_nii = nib.as_closest_canonical(
    nib.load(phantom_data_path / 'aparc.DKTatlas+aseg_native.nii.gz'))
aparc = np.pad(np.asanyarray(aparc_nii.dataobj), pad_size_220, 'constant')

roi_inds = {}
roi_inds['cortical_gm'] = np.where(aparc >= 1000)
roi_inds['eroded_cortical_gm'] = np.where(
    binary_erosion(aparc >= 1000, iterations=3))
roi_inds['putamen'] = np.where(np.isin(aparc, [12, 51]))
roi_inds['eroded_putamen'] = np.where(
    binary_erosion(np.isin(aparc, [12, 51]), iterations=5))
roi_inds['wm'] = np.where(np.isin(aparc, [2, 41]))
roi_inds['eroded_wm'] = np.where(
    binary_erosion(np.isin(aparc, [2, 41]), iterations=5))
roi_inds['ventricles'] = np.where(np.isin(aparc, [4, 43]))
roi_inds['eroded_ventricles'] = np.where(
    binary_erosion(np.isin(aparc, [4, 43]), iterations=5))
#-----------------------------------------------------------------------

# load the ground truth
gt = np.abs(np.load(odirs[0] / 'na_gt.npy'))
gt = zoom3d(gt, 440 / gt.shape[0])

norm_non_anatomical = 'L2'
norm_anatomical = 'L1'
betas_non_anatomical = [3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
betas_anatomical = [3e-3, 1e-2, 3e-2, 1e-1]

sm_fwhms_mm = [0.1, 4., 6., 8., 10.]

recons_e1_no_decay = np.zeros(
    (len(odirs), len(betas_non_anatomical), 440, 440, 440))
agrs_e1_no_decay = np.zeros((len(odirs), len(betas_anatomical), 440, 440, 440))

agrs_both_echos_w_decay = np.zeros(
    (len(odirs), len(betas_anatomical), 440, 440, 440))

ifft1s = np.zeros((len(odirs), len(sm_fwhms_mm), 440, 440, 440))

ifft_scale_fac = 1.

# calculate the ROI averages
true_means = {}
recon_e1_no_decay_roi_means = {}
agr_e1_no_decay_roi_means = {}
agr_both_echos_w_decay_roi_means = {}
ifft1_roi_means = {}

sl = 200

for i, odir in enumerate(odirs):
    print(odir)
    # load IFFT of first echo
    ifft = ifft_scale_fac * np.load(odir / 'ifft1.npy')
    ifft_voxsize = 220 / ifft.shape[0]

    for j, sm_fwhm_mm in enumerate(sm_fwhms_mm):
        ifft1s[i, j, ...] = zoom3d(
            np.abs(gaussian_filter(ifft, sm_fwhm_mm / (2.35 * ifft_voxsize))),
            440 / ifft.shape[0])

for key, inds in roi_inds.items():
    true_means[key] = gt[inds].mean()
    ifft1_roi_means[key] = np.array([[x[inds].mean() for x in y]
                                     for y in ifft1s])

ifft1s_mean = ifft1s[..., sl].mean(axis=0)
ifft1s_std = ifft1s[..., sl].std(axis=0)
ifft1s_0 = ifft1s[0, ..., sl].copy()

del ifft1s

for i, odir in enumerate(odirs):
    print(odir)
    # load iterative recons of first echo with non-anatomical prior
    for ib, beta_non_anatomical in enumerate(betas_non_anatomical):
        ofile_e1_no_decay = odir / f'recon_echo_1_no_decay_model_{norm_non_anatomical}_{beta_non_anatomical:.2E}.npz'
        d = np.load(ofile_e1_no_decay)
        recons_e1_no_decay[i, ib, ...] = zoom3d(np.abs(d['x']),
                                                440 / iter_shape[0])

for key, inds in roi_inds.items():
    recon_e1_no_decay_roi_means[key] = np.array([[x[inds].mean() for x in y]
                                                 for y in recons_e1_no_decay])

recons_e1_no_decay_mean = recons_e1_no_decay[..., sl].mean(axis=0)
recons_e1_no_decay_std = recons_e1_no_decay[..., sl].std(axis=0)
recons_e1_no_decay_0 = recons_e1_no_decay[0, ..., sl].copy()

del recons_e1_no_decay

for i, odir in enumerate(odirs):
    print(odir)
    # load AGR of first echo with out decay model
    for ib, beta_anatomical in enumerate(betas_anatomical):
        ofile_e1_no_decay_agr = odir / 'agr' / f'agr_echo_1_no_decay_model_{norm_anatomical}_{beta_anatomical:.2E}.npz'
        d = np.load(ofile_e1_no_decay_agr)
        agrs_e1_no_decay[i, ib, ...] = zoom3d(np.abs(d['x']),
                                              440 / iter_shape[0])

for key, inds in roi_inds.items():
    agr_e1_no_decay_roi_means[key] = np.array([[x[inds].mean() for x in y]
                                               for y in agrs_e1_no_decay])

agrs_e1_no_decay_mean = agrs_e1_no_decay[..., sl].mean(axis=0)
agrs_e1_no_decay_std = agrs_e1_no_decay[..., sl].std(axis=0)
agrs_e1_no_decay_0 = agrs_e1_no_decay[0, ..., sl].copy()

del agrs_e1_no_decay

for i, odir in enumerate(odirs):
    print(odir)
    # load AGR of borhs echos with decay model
    for ib, beta_anatomical in enumerate(betas_anatomical):
        ofile_both_echos_agr = odir / 'agr' / f'agr_both_echos_w_decay_model_{norm_anatomical}_{beta_anatomical:.2E}.npz'
        d = np.load(ofile_both_echos_agr)
        agrs_both_echos_w_decay[i, ib, ...] = zoom3d(np.abs(d['x']),
                                                     440 / iter_shape[0])

for key, inds in roi_inds.items():
    agr_both_echos_w_decay_roi_means[key] = np.array(
        [[x[inds].mean() for x in y] for y in agrs_both_echos_w_decay])

agrs_both_echos_w_decay_mean = agrs_both_echos_w_decay[..., sl].mean(axis=0)
agrs_both_echos_w_decay_std = agrs_both_echos_w_decay[..., sl].std(axis=0)
agrs_both_echos_w_decay_0 = agrs_both_echos_w_decay[0, ..., sl].copy()

del agrs_both_echos_w_decay

#--------------------------------------------------------------------------------
# bias noise plots

num_rows = 2
num_cols = len(roi_inds) // 2
fig, ax = plt.subplots(num_rows,
                       num_cols,
                       figsize=(4 * num_cols, 4 * num_rows))

for i, (roi, vals) in enumerate(ifft1_roi_means.items()):
    x = vals.std(0)
    y = 100 * (vals.mean(0) - true_means[roi]) / true_means[roi]
    ax.ravel()[i].plot(x, y, 'x-', label='IFFT')

    for j in range(len(x)):
        ax.ravel()[i].annotate(f'{sm_fwhms_mm[j]:.1f}mm', (x[j], y[j]),
                               horizontalalignment='center',
                               verticalalignment='bottom')

for i, (roi, vals) in enumerate(recon_e1_no_decay_roi_means.items()):
    x = vals.std(0)
    y = 100 * (vals.mean(0) - true_means[roi]) / true_means[roi]
    ax.ravel()[i].plot(x, y, 'o-', label='iter quad. prior')

    for j in range(len(x)):
        ax.ravel()[i].annotate(f'{betas_non_anatomical[j]}', (x[j], y[j]),
                               horizontalalignment='center',
                               verticalalignment='bottom')

for i, (roi, vals) in enumerate(agr_e1_no_decay_roi_means.items()):
    x = vals.std(0)
    y = 100 * (vals.mean(0) - true_means[roi]) / true_means[roi]
    ax.ravel()[i].plot(x, y, 'o-', label='AGR wo decay m.')

    for j in range(len(x)):
        ax.ravel()[i].annotate(f'{betas_anatomical[j]}', (x[j], y[j]),
                               horizontalalignment='center',
                               verticalalignment='bottom')

for i, (roi, vals) in enumerate(agr_both_echos_w_decay_roi_means.items()):
    x = vals.std(0)
    y = 100 * (vals.mean(0) - true_means[roi]) / true_means[roi]
    ax.ravel()[i].plot(x, y, 'o-', label='AGR w decay m.')

    for j in range(len(x)):
        ax.ravel()[i].annotate(f'{betas_anatomical[j]}', (x[j], y[j]),
                               horizontalalignment='center',
                               verticalalignment='bottom')

    ax.ravel()[i].set_title(roi)
    ax.ravel()[i].grid(ls=':')
    ax.ravel()[i].set_xlabel('std.dev. of ROI mean')
    ax.ravel()[i].axhline(0, color='k')

    # get y-axis limits of the plot
    low, high = ax.ravel()[i].get_ylim()
    # find the new limits
    bound = max(abs(low), abs(high))
    # set new limits
    ax.ravel()[i].set_ylim(-bound, bound)

ax[0, 0].set_ylabel('bias of ROI mean [%]')
ax[1, 0].set_ylabel('bias of ROI mean [%]')
ax.ravel()[-1].legend(loc='upper left', ncols=2)

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
    ax2[0, i].imshow(recons_e1_no_decay_0[i, ...].T,
                     origin='lower',
                     cmap='Greys_r',
                     vmin=0,
                     vmax=3.5)
    ax2[1, i].imshow(recons_e1_no_decay_mean[i, ...].T - gt[:, :, sl].T,
                     origin='lower',
                     cmap='seismic',
                     vmin=-1,
                     vmax=1)
    ax2[2, i].imshow(recons_e1_no_decay_std[i, ...].T,
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

num_cols3 = len(sm_fwhms_mm)
num_rows3 = 3
fig3, ax3 = plt.subplots(num_rows3,
                         num_cols3,
                         figsize=(3 * num_cols3, 3 * num_rows3))

for i in range(num_cols3):
    ax3[0, i].imshow(ifft1s_0[i, ...].T,
                     origin='lower',
                     cmap='Greys_r',
                     vmin=0,
                     vmax=3.5)
    ax3[1, i].imshow(ifft1s_mean[i, ...].T - gt[:, :, sl].T,
                     origin='lower',
                     cmap='seismic',
                     vmin=-1,
                     vmax=1)
    ax3[2, i].imshow(ifft1s_std[i, ...].T,
                     origin='lower',
                     cmap='Greys_r',
                     vmin=0,
                     vmax=0.5)

    ax3[0, i].set_title(f'fwhm = {sm_fwhms_mm[i]:.1f}mm')

ax3[0, 0].set_ylabel('first noise realization')
ax3[1, 0].set_ylabel('bias image')
ax3[2, 0].set_ylabel('std.dev. image')

for axx in ax3.ravel():
    axx.set_xticks([])
    axx.set_yticks([])

fig3.tight_layout()
fig3.show()

#--------------------------------------------------------------------------------

num_cols4 = len(betas_anatomical)
num_rows4 = 3
fig4, ax4 = plt.subplots(num_rows4,
                         num_cols4,
                         figsize=(3 * num_cols4, 3 * num_rows4))

for i in range(num_cols4):
    ax4[0, i].imshow(agrs_e1_no_decay_0[i, ...].T,
                     origin='lower',
                     cmap='Greys_r',
                     vmin=0,
                     vmax=3.5)
    ax4[1, i].imshow(agrs_e1_no_decay_mean[i, ...].T - gt[:, :, sl].T,
                     origin='lower',
                     cmap='seismic',
                     vmin=-1,
                     vmax=1)
    ax4[2, i].imshow(agrs_e1_no_decay_std[i, ...].T,
                     origin='lower',
                     cmap='Greys_r',
                     vmin=0,
                     vmax=0.5)

    ax4[0, i].set_title(f'beta = {betas_non_anatomical[i]}')

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
    ax5[0, i].imshow(agrs_both_echos_w_decay_0[i, ...].T,
                     origin='lower',
                     cmap='Greys_r',
                     vmin=0,
                     vmax=3.5)
    ax5[1, i].imshow(agrs_both_echos_w_decay_mean[i, ...].T - gt[:, :, sl].T,
                     origin='lower',
                     cmap='seismic',
                     vmin=-1,
                     vmax=1)
    ax5[2, i].imshow(agrs_both_echos_w_decay_std[i, ...].T,
                     origin='lower',
                     cmap='Greys_r',
                     vmin=0,
                     vmax=0.5)

    ax5[0, i].set_title(f'beta = {betas_non_anatomical[i]}')

ax5[0, 0].set_ylabel('first noise realization')
ax5[1, 0].set_ylabel('bias image')
ax5[2, 0].set_ylabel('std.dev. image')

for axx in ax5.ravel():
    axx.set_xticks([])
    axx.set_yticks([])

fig5.tight_layout()
fig5.show()

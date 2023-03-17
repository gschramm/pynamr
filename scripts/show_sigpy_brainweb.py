import argparse
import json
from pathlib import Path
import numpy as np
import pymirc.viewer as pv
import nibabel as nib
from pymirc.image_operations import zoom3d
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--max_num_iter', type=int, default=500)
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
roi_inds['putamen'] = np.where(np.isin(aparc, [12, 51]))
roi_inds['wm'] = np.where(binary_erosion(np.isin(aparc, [2, 41]),
                                         iterations=5))
roi_inds['ventricles'] = np.where(
    binary_erosion(np.isin(aparc, [4, 43]), iterations=3))
#-----------------------------------------------------------------------

# load the ground truth
gt = np.abs(np.load(odirs[0] / 'na_gt.npy'))
gt = zoom3d(gt, 440 / gt.shape[0])

norm_non_anatomical = 'L2'
betas_non_anatomical = [1e-2, 3e-2, 1e-1, 3e-1, 1e0]

recons_e1_no_decay = np.zeros(
    (len(odirs), len(betas_non_anatomical), 440, 440, 440))

for i, odir in enumerate(odirs):
    print(odir)

    for ib, beta_non_anatomical in enumerate(betas_non_anatomical):
        ofile_e1_no_decay = odir / f'recon_echo_1_no_decay_model_{norm_non_anatomical}_{beta_non_anatomical:.2E}.npz'
        d = np.load(ofile_e1_no_decay)
        recons_e1_no_decay[i, ib, ...] = zoom3d(np.abs(d['x']),
                                                440 / iter_shape[0])

# calculate the ROI averages
true_means = {}
recon_e1_no_decay_roi_means = {}

for key, inds in roi_inds.items():
    true_means[key] = gt[inds].mean()
    recon_e1_no_decay_roi_means[key] = np.array([[x[inds].mean() for x in y]
                                                 for y in recons_e1_no_decay])

#--------------------------------------------------------------------------------
# bias noise plots

num_cols = len(roi_inds)
fig, ax = plt.subplots(1, num_cols, figsize=(4 * num_cols, num_cols))

for i, (roi, vals) in enumerate(recon_e1_no_decay_roi_means.items()):
    x = vals.std(0)
    y = 100 * (vals.mean(0) - true_means[roi]) / true_means[roi]
    ax[i].plot(x, y, 'o-')

    for j in range(len(x)):
        ax[i].annotate(f'{betas_non_anatomical[j]}', (x[j], y[j]),
                       horizontalalignment='center',
                       verticalalignment='bottom')

    ax[i].set_title(roi)
    ax[i].grid(ls=':')
    ax[i].set_xlabel('std.dev. of ROI mean')
    ax[i].axhline(0, color='k')

    # get y-axis limits of the plot
    low, high = ax[i].get_ylim()
    # find the new limits
    bound = max(abs(low), abs(high))
    # set new limits
    ax[i].set_ylim(-bound, bound)

ax[0].set_ylabel('bias of ROI mean [%]')

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
    ax2[0, i].imshow(recons_e1_no_decay[0, i, :, :, 200].T,
                     origin='lower',
                     cmap='Greys_r',
                     vmin=0,
                     vmax=3.5)
    ax2[1, i].imshow(recons_e1_no_decay[:, i, :, :, 200].mean(0).T -
                     gt[:, :, 200].T,
                     origin='lower',
                     cmap='seismic',
                     vmin=-1,
                     vmax=1)
    ax2[2, i].imshow(recons_e1_no_decay[:, i, :, :, 200].std(0).T,
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
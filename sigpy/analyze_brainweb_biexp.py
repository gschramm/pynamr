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
import pandas as pd
import seaborn as sns

sns.set_context('paper')

parser = argparse.ArgumentParser()
parser.add_argument('--max_num_iter', type=int, default=2000)
parser.add_argument('--num_iter_r', type=int, default=100)
parser.add_argument('--noise_level', type=float, default=0)
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

betas_non_anatomical = [1e-1]
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
    list((Path(data_root_dir) / 'run_brainweb_biexp').glob(
        f'{phantom}_nodecay_{no_decay}_i_{max_num_iter:04}_nl_{noise_level:.1E}_s_*'
    )))

#-----------------------------------------------------------------------
# load the ground truth
gt = np.abs(np.load(odirs[0] / 'na_gt.npy'))
t1 = np.abs(np.load(odirs[0] / 't1.npy'))
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

# add the eyes ROI
x = np.linspace(0, 440 - 1, lab.shape[0])
X, Y, Z = np.meshgrid(x, x, x)
R1 = np.sqrt((X - 368)**2 + (Y - 143)**2 + (Z - 97)**2)
R2 = np.sqrt((X - 368)**2 + (Y - 291)**2 + (Z - 97)**2)
eye1_inds = np.where((R1 < 25))
eye2_inds = np.where((R2 < 25))

tmp = np.zeros_like(gt, dtype=np.uint8)
tmp[eye1_inds] = 1
tmp[eye2_inds] = 1

roi_inds['eyes'] = np.where(tmp * (np.abs(gt - 1.5) < 0.01))

# add lesion ROI
R1 = np.sqrt((X - 329)**2 + (Y - 165)**2 + (Z - 200)**2)
roi_inds['lesion'] = np.where((R1 < 10) * (np.abs(gt - 0.6) < 0.01))

roi_inds['white matter'] = np.where(np.isin((aparc * wm_mask), [2, 41]))

roi_inds['putamen'] = np.where(np.isin((aparc * gm_mask), [12, 51]))
roi_inds['caudate'] = np.where(np.isin((aparc * gm_mask), [11, 50]))
roi_inds['cerebellum'] = np.where(np.isin((aparc * gm_mask), [8, 47]))
roi_inds['cortical grey matter'] = np.where((aparc * gm_mask) >= 1000)
roi_inds['frontal'] = np.where(np.isin((aparc * gm_mask), [1028, 2028]))
roi_inds['temporal'] = np.where(
    np.isin((aparc * gm_mask), [1009, 1015, 1030, 2009, 2015, 2030]))

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

for i, odir in enumerate(odirs):
    print('loading AGR w decay model', odir)
    # load AGR of borhs echos with decay model
    for ib, beta_anatomical in enumerate(betas_anatomical):
        ofile_both_echos_agr0 = odir / f'agr_both_echos_w_biexpdecay_model_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{max_num_iter}.npz'
        d0 = np.load(ofile_both_echos_agr0)

        agrs_both_echos_w_decay0[i, ib,
                                 ...] = zoom(np.abs(d0['x'] / image_scale),
                                             sim_shape[0] / iter_shape[0],
                                             order=1,
                                             prefilter=False)

for key, inds in roi_inds.items():
    agr_both_echos_w_decay0_roi_means[key] = np.array(
        [[x[inds].mean() for x in y] for y in agrs_both_echos_w_decay0])

agrs_both_echos_w_decay0_mean = agrs_both_echos_w_decay0.mean(axis=0)

for i, (roi, vals) in enumerate(agr_both_echos_w_decay0_roi_means.items()):
    y = 100 * (vals.mean(0) - true_means[roi]) / true_means[roi]
    print(roi, y)
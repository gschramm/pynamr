import argparse
from pathlib import Path
import numpy as np
import pymirc.viewer as pv
from pymirc.image_operations import zoom3d
from scipy.ndimage import binary_erosion

parser = argparse.ArgumentParser()
parser.add_argument('--regularization_norm',
                    type=str,
                    default='L1',
                    choices=['L1', 'L2'])
parser.add_argument('--beta', type=float, default=2e-2)
parser.add_argument('--max_num_iter', type=int, default=300)
parser.add_argument('--noise_level', type=float, default=3e-2)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

regularization_norm = args.regularization_norm
beta = args.beta
max_num_iter = args.max_num_iter
noise_level = args.noise_level
seed = args.seed

regularization_norm_non_anatomical = 'L2'
beta_non_anatomical = 2e-1
short_fraction = 0.6
ishape = (128, 128, 128)

zoomfac = 256 / ishape[0]

#---------------------------------------------------------------

odir = Path('run') / f'i_{max_num_iter:04}_nl_{noise_level:.1E}_s_{seed:03}'

x = np.abs(np.load(odir / 'na_gt.npy'))
t1_image = np.load(odir / 't1.npy')
true_ratio_image_short = np.load(odir / 'true_ratio_short.npy')
true_ratio_image_long = np.load(odir / 'true_ratio_long.npy')

x = zoom3d(x, 256 / x.shape[0])
t1_image = zoom3d(t1_image, 256 / t1_image.shape[0])
true_ratio_image_short = zoom3d(true_ratio_image_short,
                                256 / true_ratio_image_short.shape[0])
true_ratio_image_long = zoom3d(true_ratio_image_long,
                               256 / true_ratio_image_long.shape[0])

ifft1 = zoom3d(np.abs(np.load(odir / 'ifft1.npy')), zoomfac)
ifft2 = zoom3d(np.abs(np.load(odir / 'ifft2.npy')), zoomfac)
ifft1_filt = zoom3d(np.abs(np.load(odir / 'ifft1_filt.npy')), zoomfac)
ifft2_filt = zoom3d(np.abs(np.load(odir / 'ifft2_filt.npy')), zoomfac)

outfile1 = odir / f'recon_echo_1_no_decay_model_{regularization_norm_non_anatomical}_{beta_non_anatomical:.1E}.npz'
d1 = np.load(outfile1)
recon_echo_1_wo_decay_model = zoom3d(np.abs(d1['x']), zoomfac)

outfile2 = odir / f'recon_echo_2_no_decay_model_{regularization_norm_non_anatomical}_{beta_non_anatomical:.1E}.npz'
d2 = np.load(outfile2)
recon_echo_2_wo_decay_model = zoom3d(np.abs(d2['x']), zoomfac)

outfile3 = odir / f'agr_echo_1_no_decay_model_{regularization_norm}_{beta:.1E}.npz'
d3 = np.load(outfile3)
agr_echo_1_wo_decay_model = zoom3d(np.abs(d3['x']), zoomfac)

outfile4 = odir / f'agr_echo_2_no_decay_model_{regularization_norm}_{beta:.1E}.npz'
d4 = np.load(outfile4)
agr_echo_2_wo_decay_model = zoom3d(np.abs(d4['x']), zoomfac)

outfile5 = odir / f'agr_echo_1_true_decay_model_{regularization_norm}_{beta:.1E}.npz'
d5 = np.load(outfile5)
agr_echo_1_true_decay_model = zoom3d(np.abs(d5['x']), zoomfac)

outfile6 = odir / f'agr_echo_2_true_decay_model_{regularization_norm}_{beta:.1E}.npz'
d6 = np.load(outfile6)
agr_echo_2_true_decay_model = zoom3d(np.abs(d6['x']), zoomfac)

outfile7 = odir / f'agr_echo_1_est_decay_model_{regularization_norm}_{beta:.1E}.npz'
d7 = np.load(outfile7)
agr_echo_1_est_decay_model = zoom3d(np.abs(d7['x']), zoomfac)

outfile8 = odir / f'agr_echo_2_est_decay_model_{regularization_norm}_{beta:.1E}.npz'
d8 = np.load(outfile8)
agr_echo_2_est_decay_model = zoom3d(np.abs(d8['x']), zoomfac)

est_ratio = zoom3d(np.load(odir / 'est_ratio.npy'), zoomfac)

#--------------------------------------------------------------------------------------
gm = np.load(odir.parent / 'gm_256.npy')
wm = np.load(odir.parent / 'wm_256.npy')

wm_mask = binary_erosion(np.abs(x * wm - 1) < 0.01, iterations=2)

wm_inds = np.where(wm_mask)
gm_inds = np.where(gm)

gmwm_ratio_non_anatomical = recon_echo_1_wo_decay_model[gm_inds].mean(
) / recon_echo_1_wo_decay_model[wm_inds].mean()

gmwm_ratio_anatomical_wo_decay_model = agr_echo_1_wo_decay_model[gm_inds].mean(
) / agr_echo_1_wo_decay_model[wm_inds].mean()

gmwm_ratio_anatomical_true_decay_model = agr_echo_1_true_decay_model[
    gm_inds].mean() / agr_echo_1_true_decay_model[wm_inds].mean()

gmwm_ratio_anatomical_est_decay_model = agr_echo_1_est_decay_model[
    gm_inds].mean() / agr_echo_1_est_decay_model[wm_inds].mean()

gmwm_ratio_true = x[gm_inds].mean() / x[wm_inds].mean()

#--------------------------------------------------------------------------------------
# approximate "true" monoexp. ratio
true_ratio = short_fraction * true_ratio_image_short + (
    1 - short_fraction) * true_ratio_image_long

ims = 3 * [dict(vmin=0, vmax=3.5, cmap='Greys_r')]
kwargs = dict(sl_z=127)

vi1 = pv.ThreeAxisViewer(
    [
        np.flip(x, (0, 1)) for x in [
            np.abs(ifft1),
            np.abs(ifft1_filt),
            np.abs(recon_echo_1_wo_decay_model),
        ]
    ],
    imshow_kwargs=ims,
    rowlabels=['', '', f'GM/WM = {gmwm_ratio_non_anatomical:.2f}'],
    **kwargs)

vi2 = pv.ThreeAxisViewer(
    [
        np.flip(x, (0, 1)) for x in [
            np.abs(agr_echo_1_wo_decay_model),
            np.abs(agr_echo_1_true_decay_model),
            np.abs(agr_echo_1_est_decay_model),
        ]
    ],
    imshow_kwargs=ims,
    rowlabels=[
        f'GM/WM = {gmwm_ratio_anatomical_wo_decay_model:.2f}',
        f'GM/WM = {gmwm_ratio_anatomical_true_decay_model:.2f}',
        f'GM/WM = {gmwm_ratio_anatomical_est_decay_model:.2f}'
    ],
    **kwargs)

vi3 = pv.ThreeAxisViewer([
    np.flip(x, (0, 1)) for x in [
        np.abs(ifft2),
        np.abs(ifft2_filt),
        np.abs(recon_echo_2_wo_decay_model),
    ]
],
                         imshow_kwargs=ims,
                         **kwargs)

vi4 = pv.ThreeAxisViewer([
    np.flip(x, (0, 1)) for x in [
        np.abs(agr_echo_2_wo_decay_model),
        np.abs(agr_echo_2_true_decay_model),
        np.abs(agr_echo_2_est_decay_model),
    ]
],
                         imshow_kwargs=ims,
                         **kwargs)

vi5 = pv.ThreeAxisViewer([np.flip(x, (0, 1)) for x in [true_ratio, est_ratio]],
                         imshow_kwargs=dict(vmin=0, vmax=1, cmap='Greys_r'),
                         **kwargs)

vi6 = pv.ThreeAxisViewer([np.flip(x, (0, 1)) for x in [x, 0.003 * t1_image]],
                         imshow_kwargs=dict(vmin=0, vmax=3.5, cmap='Greys_r'),
                         rowlabels=[f'GM/WM = {gmwm_ratio_true:.2f}', ''],
                         **kwargs)

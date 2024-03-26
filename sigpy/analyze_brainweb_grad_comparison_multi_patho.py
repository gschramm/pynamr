""" Analyze results from the comparison of different TPI max gradients (brainweb sim + recon):
    - ROI bias-stddev and SNR/CNR in image space for glioma and cosine lesion
    - several pathologies in the same image to reduce the number of recons,
    though it adds some complications (e.g. the ideal observer analysis requires a single pathology)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import json
import time
import re as re
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.signal import find_peaks
from skimage.morphology import binary_dilation
import seaborn as sns
import sys
import pymirc.viewer as pv
from pymirc.image_operations import zoom3d
import argparse
from numpy.random import default_rng
from utils import ideal_observer_snr

#------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    '--recon_type',
    type=str,
    default='decay_model_iter',
    choices=[
        'simple_iter', 'decay_model_iter', 'highres_prior_decay_model_iter',
        'highres_prior_simple_iter'
    ])
parser.add_argument('--load_baseline_realiz', action='store_true')
parser.add_argument('--no_decay', action='store_true')
parser.add_argument('--report_mode', action='store_true')
parser.add_argument('--nb_seeds', type=int, default=100)
parser.add_argument(
    '--pathologies',
    type=str,
    default='glioma+lesion_gwm_cos',
    choices=[
        'glioma+lesion_gwm_cos', 'glioma_treatment+lesion_gwm_cos',
        'lesion1.6+lesion3+lesion5', 'lesion3.2+lesion4.6+lesion6.6'
    ])
parser.add_argument(
    '--baseline_pathologies',
    type=str,
    default='none',
    choices=[
        'glioma+lesion_gwm_cos',  # for glioma_treatment+lesion_gwm_cos, though glioma_treatment+none would be better/cleaner
        'lesion3.2+lesion4.6+lesion6.6'  # for lesion1.6+lesion3+lesion5
    ])

args = parser.parse_args()

recon_type = args.recon_type
load_baseline_realiz = args.load_baseline_realiz
no_decay = args.no_decay
report_mode = args.report_mode
nb_seeds = args.nb_seeds
pathologies = args.pathologies
baseline_pathologies = args.baseline_pathologies

#------------------------------------------------------------------------------
# setup


# helper function for computing Z (last dimension) profile only in input ROI:
# average the 3D input image over the first two dimensions (XY)
# only in the ROI specified by the input mask
def profile_z_roi(im: np.ndarray, mask: np.ndarray):
    # mean only inside mask, results in NaN outside of mask over Z
    profile_z = np.mean(im, axis=(0, 1), where=mask)
    # extract only meaningful signal (discard NaNs)
    profile_z = profile_z[np.logical_not(np.isnan(profile_z))]
    return profile_z


plt.ion()

start_time = time.time()

# recon names
if recon_type == 'simple_iter':
    recon_desc = '_bfgs'
elif recon_type == 'decay_model_iter':
    recon_desc = '_bfgs_true_biexp'
elif recon_type == 'highres_prior_decay_model_iter':
    recon_desc = '_bfgs_highres_prior_true_biexp'
elif recon_type == 'highres_prior_simple_iter':
    recon_desc = '_bfgs_highres_prior'

# individual pathologies
sub_patho = str.split(pathologies, '+')
if baseline_pathologies == 'none':
    sub_patho_baseline = ['none' for s in range(len(sub_patho))]
else:
    sub_patho_baseline = str.split(baseline_pathologies, '+')
    assert (len(sub_patho) == len(sub_patho_baseline))

# noise level to the max/DC in k-space
noise_level = 1.4e-2

# reconstruction grid
recon_n = 128

# directories
# read the data root directory from the config file
with open('.simulation_config.json', 'r') as f:
    data_root_dir: str = json.load(f)['data_root_dir']
data_root_dir = Path(data_root_dir)

save_dir = data_root_dir / f"analysis_final_brainweb_{pathologies}_grad_recon_{recon_type}_baseline_{baseline_pathologies}_seeds{nb_seeds}"
if not save_dir.exists():
    save_dir.mkdir(exist_ok=True, parents=True)

recon_path: Path = data_root_dir / f"final_recon_brainweb_patho_{pathologies}_noise{noise_level:.1E}{'_no_decay' if no_decay else ''}"
recon_path_baseline: Path = data_root_dir / f"final_recon_brainweb_patho_{baseline_pathologies}_noise{noise_level:.1E}{'_no_decay' if no_decay else ''}"
sim_path: Path = data_root_dir / f"final_sim_brainweb_patho_{pathologies}{'_no_decay' if no_decay else ''}"
sim_path_baseline: Path = data_root_dir / f"final_sim_brainweb_patho_{baseline_pathologies}{'_no_decay' if no_decay else ''}"
phantom_path: Path = data_root_dir / 'brainweb54'

# lists of parameter values
grads = [16, 32, 48]

# regularization strength
betas = [1e-3, 5e-3, 1e-2, 5e-2]

# realization IDs (seeds)
rng = default_rng(42)
seeds = rng.integers(1, 2**31 - 1, size=100)
seeds = seeds[:nb_seeds]

nb_grad = len(grads)
nb_beta = len(betas)
nb_realiz = len(seeds)
print('gradients')
print(grads)
print('beta')
print(betas)
print('seeds')
print(seeds)

# example realization for display
example_realiz = seeds[0]

# analysis: parameterization of reconstructed images
params = ['grad', 'beta', 'recon_type']

# analysis: rois setup
# general_rois = ['wm', 'gm', 'csf']
if pathologies == 'glioma+lesion_gwm_cos' or pathologies == 'glioma_treatment+lesion_gwm_cos':
    glioma_patho = sub_patho[0]
    # rois for which the contrast wrt to ref wm and baseline should be computed
    rois_for_contrast = [glioma_patho + '_ring', glioma_patho + '_core']
    # renaming for nicer final display, applied to final metrics
    roi_rename_dict = {'treatment': 'trtm'}
elif pathologies == 'lesion1.6+lesion3+lesion5' or pathologies == 'lesion3.2+lesion4.6+lesion6.6':
    rois_for_contrast = sub_patho.copy()
    # renaming from matrix size percentage to mm
    roi_rename_dict = {
        'lesion1\.6(?=\s?_?)': 'lesion3.4',
        'lesion3(?=\s?_?[^\.])': 'lesion6',
        '^lesion3$': 'lesion6',
        'lesion5(?=\s?_?)': 'lesion11'
    }

# all ROIs for bias/std/snr of ROI mean
# the ROIs are not necessarily identical to individual pathology masks
# the same ROIs are used for the baseline images if loaded
rois_all = rois_for_contrast + ['ref_wm']

#------------------------------------------------------------------------------
# load images and masks needed for analysis

# load true simulated na images, saved as complex but actually real,
# oriented for the pymirc 3Dviewer
true_na_image = np.load(sim_path / 'na_gt.npy').real
true_na_image_baseline = np.load(sim_path_baseline / 'na_gt.npy').real

# simulation grid, cubic
sim_shape = true_na_image.shape
sim_n = sim_shape[0]
zoom_factor = sim_n // recon_n

# load the masks (nifti oriented)
masks = dict()
for r in rois_all:
    masks[r] = np.load(phantom_path / f'{r}_{sim_n}.npy')[::-1, ::-1]

# load also noiseless recons, baseline noiseless recons
recon_noiseless_path: Path = data_root_dir / f"final_recon_brainweb_patho_{pathologies}_noise{0.:.1E}{'_no_decay' if no_decay else ''}"
recon_noiseless = np.zeros((nb_grad, nb_beta, *sim_shape), np.float64)
recon_noiseless_path_baseline: Path = data_root_dir / f"final_recon_brainweb_patho_{baseline_pathologies}_noise{0.:.1E}{'_no_decay' if no_decay else ''}"
recon_noiseless_baseline = np.zeros((nb_grad, nb_beta, *sim_shape), np.float64)
for g, grad in enumerate(grads):
    for b, beta in enumerate(betas):
        path = recon_noiseless_path / f'recon_te1{recon_desc}_beta{beta:.1E}_it200_grad{grad}_seed191664964.npz'
        recon = np.abs(np.load(path)['recon'])
        # for the quantification interpolate the reconstructed
        # image to the simulation grid
        recon_interp = zoom3d(recon, zoom_factor)
        recon_noiseless[g, b] = recon_interp
        # no patho
        path = recon_noiseless_path_baseline / f'recon_te1{recon_desc}_beta{beta:.1E}_it200_grad{grad}_seed191664964.npz'
        recon = np.abs(np.load(path)['recon'])
        # for the quantification interpolate the reconstructed
        # image to the simulation grid
        recon_interp = zoom3d(recon, zoom_factor)
        recon_noiseless_baseline[g, b] = recon_interp

#wm_mask = np.load(phantom_path / f'wm_{sim_n}.npy')[::-1, ::-1]
#csf_mask = np.load(phantom_path / f'csf_{sim_n}.npy')[::-1, ::-1]
#cortex_mask = np.load(phantom_path / f'cortex_{sim_n}.npy')[::-1, ::-1]
#glioma_core_mask = np.load(
#    phantom_path / f'glioma_{sim_n}_core.npy')[::-1, ::-1]
#glioma_ring_mask = np.load(
#    phantom_path / f'glioma_{sim_n}_ring.npy')[::-1, ::-1]
#glioma_ring_contralateral_mask = glioma_ring_mask[::-1,:,:] * wm_mask
#glioma_ring_contralateral_mask = binary_erosion(glioma_ring_contralateral_mask)

# lesion in GWM with cosine a bit more tricky for analysis
# entire mask, GWM and WM version
# isolated pathology, as pathological - baseline
if 'cos' in pathologies:
    true_cos_gwm_diff = np.load(sim_path / 'lesion_gwm_cos_diff.npy').real
    wm_mask = np.load(phantom_path / f'wm_{sim_n}.npy')[::-1, ::-1]
    cos_gwm_mask = np.abs(true_cos_gwm_diff) > 0.
    cos_wm_mask = cos_gwm_mask * wm_mask
    # as cos lesion intensity uniform in transaxial plane (XY),
    # analyze Z profiles after summing over XY
    true_cos_wm_prof_z_diff = profile_z_roi(true_cos_gwm_diff, cos_wm_mask)
    true_cos_wm_prof_z = profile_z_roi(true_na_image, cos_wm_mask)
    # cosine amplitude metric
    cos_wm_peaks = find_peaks(true_cos_wm_prof_z_diff)[0][1:3]
    cos_wm_valleys = find_peaks(-true_cos_wm_prof_z_diff)[0][1:3]

# exclude subcortical region from GM mask
#gm_mask[cortex_mask == 0] = 0
#gm_mask_dilated = binary_dilation(gm_mask, iterations=5).astype(int)
#wm_local_mask = np.logical_and((gm_mask_dilated - gm_mask).astype(int),
#                               wm_mask.astype(int))

# image space naive observer templates
diff_noiseless = recon_noiseless - recon_noiseless_baseline
im_obs_masks = dict()
im_obs_template = dict()
for p_ind, p in enumerate(sub_patho):
    # the mask is not the entire image because we put several independent pathologies into the same image
    # for speeding up computations
    im_obs_masks[p] = np.load(
        phantom_path / f'im_obs_{p}_{sim_n}.npy')[::-1, ::-1]
    # image observer masks is the union of pathology and its baseline (may differ in some cases)
    if sub_patho_baseline[p_ind] != 'none':
        im_obs_masks[p] = np.logical_or(
            im_obs_masks[p],
            np.load(
                phantom_path /
                f'im_obs_{sub_patho_baseline[p_ind]}_{sim_n}.npy')[::-1, ::-1])
    im_obs_template[p] = np.multiply(diff_noiseless, im_obs_masks[p])

# metrics on the true image
true = dict()
for r in rois_all:
    true[r] = true_na_image[masks[r]].mean()
    if load_baseline_realiz:
        true[r + '_baseline'] = true_na_image_baseline[masks[r]].mean()

for r in rois_for_contrast:
    true[r + '_contrast_wm'] = true_na_image[masks[r]].mean() - true_na_image[
        masks['ref_wm']].mean()
    if load_baseline_realiz:
        true[r + '_contrast_baseline'] = true[r] - true_na_image_baseline[
            masks[r]].mean()

# cosine lesion special case
if 'cos' in pathologies:
    true['cos_wm_ampl'] = np.mean(true_cos_wm_prof_z[cos_wm_peaks]) - np.mean(
        true_cos_wm_prof_z[cos_wm_valleys])

#------------------------------------------------------------------------------
# read actual recons, compute and store metrics into dataframe
# and into additional vars
recon_example = np.zeros((nb_grad, nb_beta, *sim_shape), np.float64)
recon_realiz_mean = np.zeros((nb_grad, nb_beta, *sim_shape), np.float64)
recon_realiz_var = np.zeros((nb_grad, nb_beta, *sim_shape), np.float64)
if load_baseline_realiz:
    recon_realiz_mean_baseline = np.zeros((nb_grad, nb_beta, *sim_shape),
                                          np.float64)
    recon_realiz_var_baseline = np.zeros((nb_grad, nb_beta, *sim_shape),
                                         np.float64)

# cosine lesion special case
if 'cos' in pathologies:
    recon_cos_wm_prof_z = np.zeros(
        (nb_realiz, nb_grad, nb_beta, np.size(true_cos_wm_prof_z_diff)),
        np.float64)

df_list = []
m = 0
# load images, compute metrics and store results into the database
# too many images for storing all the realizations into an array
for b, beta in enumerate(betas):
    for g, grad in enumerate(grads):
        for s, seed in enumerate(seeds):
            temp = pd.DataFrame(index=[m])

            # iterative recon with decay model
            temp['recon_type'] = recon_type
            temp['grad'] = grad
            # noise realization
            temp['seed'] = seed
            # regularization beta
            temp['beta'] = beta

            file_name = f'recon_te1{recon_desc}_beta{beta:.1E}_it200_grad{grad}_seed{seed}.npz'
            recon_file = recon_path / file_name

            # work with magnitude images
            recon = np.abs(np.load(recon_file)['recon'])
            # for the quantification interpolate the reconstructed
            # image to the simulation grid
            recon_interp = zoom3d(recon, zoom_factor)
            if load_baseline_realiz:
                recon_file_baseline = recon_path_baseline / file_name
                recon_baseline = np.abs(np.load(recon_file_baseline)['recon'])
                recon_interp_baseline = zoom3d(recon_baseline, zoom_factor)

            # single example realization
            if seed == example_realiz:
                recon_example[g, b] = recon_interp

            # compute the mean over realizations
            recon_realiz_mean[g, b] += recon_interp
            recon_realiz_var[g, b] += np.multiply(recon_interp, recon_interp)
            if load_baseline_realiz:
                recon_realiz_mean_baseline[g, b] += recon_interp_baseline
                recon_realiz_var_baseline[g, b] += np.multiply(
                    recon_interp_baseline, recon_interp_baseline)

            # compute ROI means
            for roi in rois_all:
                roi_mean = recon_interp[masks[roi]].mean()
                temp[roi] = float(roi_mean)
                if load_baseline_realiz and roi != 'ref_wm':
                    roi_mean_baseline = recon_interp_baseline[
                        masks[roi]].mean()
                    temp[roi + '_baseline'] = float(roi_mean_baseline)

            # compute contrast wrt to normal ref wm and baseline if loaded
            # the ROI masks are identical for the current ROI and its baseline
            ref_wm_roi_mean = recon_interp[masks['ref_wm']].mean()
            for roi in rois_for_contrast:
                roi_mean = recon_interp[masks[roi]].mean()
                temp[roi + '_contrast_wm'] = float(roi_mean - ref_wm_roi_mean)
                if load_baseline_realiz:
                    roi_mean_baseline = recon_interp_baseline[
                        masks[roi]].mean()
                    temp[roi + '_contrast_baseline'] = float(roi_mean -
                                                             roi_mean_baseline)

            # naive image observer
            for p in sub_patho:
                temp[p + '_im_obs'] = np.sum(
                    np.multiply(im_obs_template[p][g, b][im_obs_masks[p]],
                                recon_interp[im_obs_masks[p]]))

            # cosine special case
            if 'cos' in pathologies:
                cos_wm_prof_z = profile_z_roi(recon_interp, cos_wm_mask)
                temp['cos_wm_ampl'] = np.mean(
                    cos_wm_prof_z[cos_wm_peaks]) - np.mean(
                        cos_wm_prof_z[cos_wm_valleys])

                # save realization for voxels belonging to the cosine lesion
                recon_cos_wm_prof_z[s, g, b] = cos_wm_prof_z

            # add to database
            df_list.append(temp)

            # display progress
            m += 1
            print(
                f'processed {m}th image out of {nb_grad*nb_beta*nb_realiz}',
                end='\r')

        # finish mean over realiz and var after the loop over seeds
        recon_realiz_mean[g, b] /= nb_realiz
        recon_realiz_var[g, b] /= nb_realiz
        recon_realiz_var[g, b] -= np.multiply(recon_realiz_mean[g, b],
                                              recon_realiz_mean[g, b])
        if load_baseline_realiz:
            recon_realiz_mean_baseline[g, b] /= nb_realiz
            recon_realiz_var_baseline[g, b] /= nb_realiz
            recon_realiz_var_baseline[g, b] -= np.multiply(
                recon_realiz_mean_baseline[g, b],
                recon_realiz_mean_baseline[g, b])

# build the entire database
df = pd.concat(df_list)

# sort
df.sort_values(['beta', 'grad', 'seed', 'recon_type'], inplace=True)
df.reset_index(inplace=True, drop=True)

# convert colums to categorical
cats = ['grad', 'beta', 'seed', 'recon_type']
df[cats] = df[cats].astype('category')

# there should be a single entry for each possible combination of categories
assert (df.shape[0] == nb_grad * nb_beta * nb_realiz)

#------------------------------------------------------------------------------
# visualize recons over param values

# visualization min/max and color map
ims_na = dict(
    vmax=true_na_image.max(), vmin=true_na_image.min(), cmap=plt.cm.viridis)
ims_t1 = dict(cmap=plt.cm.gray)
var_na = dict(cmap=plt.cm.viridis)


# show an overview of images over recon/acquisition parameters
def display_overview_2D(images_over_params,
                        dims_info,
                        title,
                        display_params={}):
    # vals_in_chunk param values per figure,
    # possibly several figures
    na_figs = []
    na_axs = []
    vals_in_chunk = 4
    nb_chunks = int(np.ceil(images_over_params.shape[1] / vals_in_chunk))
    for c in range(nb_chunks):
        temp_fig, temp_ax = plt.subplots(
            images_over_params.shape[0], vals_in_chunk, figsize=(9, 9))
        na_figs.append(temp_fig)
        na_axs.append(temp_ax)

    for fig_index in range(nb_chunks):
        # fig title
        na_figs[fig_index].suptitle(
            f'{title} (chunk {fig_index})', fontsize='small')
        # rows over first parameter
        for row_index in range(images_over_params.shape[0]):
            # cols over second parameter
            for col_index in range(vals_in_chunk):
                val_index = fig_index * vals_in_chunk + col_index
                if val_index >= images_over_params.shape[1]:
                    continue
                # reorienting from pymirc viewer 3D orientation to 2D matplotlib
                na_axs[fig_index][row_index, col_index].imshow(
                    images_over_params[row_index, val_index], **display_params)

                if row_index == 0:
                    na_axs[fig_index][row_index, col_index].set_title(
                        f"{dims_info[1]['name']} {dims_info[1]['values'][val_index]}",
                        fontsize='small')

                if col_index == 0:
                    na_axs[fig_index][row_index, col_index].set_ylabel(
                        f"{dims_info[0]['name']} {dims_info[0]['values'][row_index]}",
                        fontsize='small')

                # formatting for axis
                na_axs[fig_index][row_index, col_index].tick_params(
                    left=False,
                    right=False,
                    labelleft=False,
                    labelbottom=False,
                    bottom=False)

        # save fig
        plt.savefig(save_dir / f'multiparam_overview_{title}.png')


# display info
dims_info = [{
    'name': 'grad',
    'values': grads
}, {
    'name': 'beta',
    'values': betas
}]

# use data for a single realization
df_viz = df.loc[df.seed == example_realiz]

# config for specific pathologies
# reorientation for matplotlib 2D display
if 'glioma' in pathologies and 'cos' in pathologies:
    slice_z = 131
    slice_y = 130
    true_na_image_viz = [
        true_na_image[:, :, slice_z].T, true_na_image[:, slice_y, ::-1].T
    ]
    recon_noiseless_viz = [
        np.transpose(recon_noiseless[:, 0, :, :, slice_z], (0, 2, 1)),
        np.transpose(recon_noiseless[:, 0, :, slice_y, ::-1], (0, 2, 1))
    ]
    images_over_params_example = [
        np.transpose(recon_example[..., slice_z], (0, 1, 3, 2)),
        np.transpose(recon_example[..., slice_y, ::-1], (0, 1, 3, 2))
    ]
    images_over_params_realiz_mean = [
        np.transpose(recon_realiz_mean[..., slice_z], (0, 1, 3, 2)),
        np.transpose(recon_realiz_mean[..., slice_y, ::-1], (0, 1, 3, 2))
    ]
    images_over_params_realiz_var = [
        np.transpose(recon_realiz_var[..., slice_z], (0, 1, 3, 2)),
        np.transpose(recon_realiz_var[..., slice_y, ::-1], (0, 1, 3, 2))
    ]
    images_over_params_noiseless = [
        np.transpose(recon_noiseless[..., slice_z], (0, 1, 3, 2)),
        np.transpose(recon_noiseless[..., slice_y, ::-1], (0, 1, 3, 2))
    ]

elif pathologies == 'lesion1.6+lesion3+lesion5' or pathologies == 'lesion3.2+lesion4.6+lesion6.6':
    slice_z = [135, 153, 143]
    true_na_image_viz = [true_na_image[:, :, sl].T for sl in slice_z]
    recon_noiseless_viz = [
        np.transpose(recon_noiseless[:, 0, :, :, sl], (0, 2, 1))
        for sl in slice_z
    ]
    images_over_params_example = [
        np.transpose(recon_example[..., sl], (0, 1, 3, 2)) for sl in slice_z
    ]
    images_over_params_realiz_mean = [
        np.transpose(recon_realiz_mean[..., sl], (0, 1, 3, 2))
        for sl in slice_z
    ]
    images_over_params_realiz_var = [
        np.transpose(recon_realiz_var[..., sl], (0, 1, 3, 2)) for sl in slice_z
    ]
    images_over_params_noiseless = [
        np.transpose(recon_noiseless[..., sl], (0, 1, 3, 2)) for sl in slice_z
    ]

# show the truth
truth_fig, truth_ax = plt.subplots(4, len(sub_patho), figsize=(9, 9))
# reorienting from pymirc viewer 3D orientation to 2D matplotlib
for a, t in enumerate(true_na_image_viz):
    truth_ax[0][a].imshow(t, **ims_na)
for g in range(len(grads)):
    for a in range(len(sub_patho)):
        truth_ax[g + 1][a].imshow(recon_noiseless_viz[a][g], **ims_na)
for i in range(len(truth_ax)):
    for j in range(len(truth_ax[i])):
        truth_ax[i][j].axis('off')
truth_fig.tight_layout()
truth_fig.suptitle('Truth \n noiseless recon', fontsize='small')
plt.savefig(save_dir / 'truth_noiseless_recon.png', bbox_inches='tight')

truth_fig, truth_ax = plt.subplots(1, len(sub_patho), figsize=(9, 4))
for a, t in enumerate(true_na_image_viz):
    truth_ax[a].imshow(t, **ims_na)
    truth_ax[a].axis('off')
plt.savefig(save_dir / f'truth_patho_{pathologies}.png', bbox_inches='tight')

for a, t in enumerate(true_na_image_viz):
    truth_fig, truth_ax = plt.subplots(figsize=(4, 4))
    truth_ax.imshow(t, **ims_na)
    truth_ax.axis('off')
    #    # arrow for lesion3
    #    plt.arrow(123,123,15,-7, width=0.8, head_width = 5, color='r')
    plt.savefig(
        save_dir / f'truth_patho_{sub_patho[a]}.png', bbox_inches='tight')

# display images over params
for a in range(len(sub_patho)):
    display_overview_2D(images_over_params_example[a], dims_info,
                        f'{sub_patho[a]}_example', ims_na)
    display_overview_2D(images_over_params_realiz_mean[a], dims_info,
                        f'{sub_patho[a]}_realiz_mean', ims_na)
    display_overview_2D(images_over_params_realiz_var[a], dims_info,
                        f'{sub_patho[a]}_realiz_var')
    display_overview_2D(images_over_params_noiseless[a], dims_info,
                        f'{sub_patho[a]}_noiseless', ims_na)

# ----------------------------------------------------------------
# visualize cos wm lesion Z profiles
if 'cos' in pathologies:
    prof_mean = np.mean(recon_cos_wm_prof_z, axis=0)
    prof_std = np.std(recon_cos_wm_prof_z, axis=0)

    fig, ax = plt.subplots()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors_grad = prop_cycle.by_key()['color']
    for b in range(nb_beta):
        for g in range(nb_grad):
            if b == 0:
                label = grads[g]
            else:
                label = ''
            ax.plot(prof_mean[g, b], color=colors_grad[g], label=label)
    ax.plot(true_cos_wm_prof_z, color='k', label='true')
    for p in cos_wm_peaks:
        ax.axvline(p, color='tab:gray')
    for p in cos_wm_valleys:
        ax.axvline(p, color='tab:gray')
    fig.suptitle('cos wm Z profile\nmean over realizations', fontsize='small')
    ax.legend()
    plt.savefig(save_dir / f'cos_wm_prof_z.pdf', bbox_inches='tight')

# ----------------------------------------------------------------
# display illustrations for the report

v = []
# example realization
ims = [
    true_na_image, recon_example[0, 1], recon_example[1, 1],
    recon_example[2, 1]
]
v.append(
    pv.ThreeAxisViewer(
        ims, imshow_kwargs=ims_na, rowlabels=['', '', '', ''], ls=''))
plt.savefig(
    save_dir / f'brainweb{recon_desc}_recons_grads_annot.png',
    bbox_inches='tight')
# noiseless
ims = [
    true_na_image, recon_noiseless[0, 1], recon_noiseless[1, 1],
    recon_noiseless[2, 1]
]
v.append(
    pv.ThreeAxisViewer(
        ims,
        imshow_kwargs=ims_na,
        rowlabels=['True', 'Grad 0.16', 'Grad 0.32', 'Grad 0.48'],
        ls=''))
plt.savefig(
    save_dir / f'brainweb{recon_desc}_noiseless_recons_grads_annot.png',
    bbox_inches='tight')

## adjoint recon
#recon_adjoint_noiseless = np.zeros((nb_grad, *true_na_image.shape), np.float64)
#recon_adjoint = np.zeros((nb_grad, *true_na_image.shape), np.float64)
#for g, grad in enumerate(grads):
#    path = recon_noiseless_path / f'recon_te1_adjoint_grad{grad}_seed191664964.npz'
#    recon = np.abs(np.load(path)['recon'])
#    # for the quantification interpolate the reconstructed
#    # image to the simulation grid
#    recon_interp = zoom3d(recon, zoom_factor)
#    recon_adjoint_noiseless[g] = recon_interp
#
#    path = recon_path / f'recon_te1_adjoint_grad{grad}_seed191664964.npz'
#    recon = np.abs(np.load(path)['recon'])
#    # for the quantification interpolate the reconstructed
#    # image to the simulation grid
#    recon_interp = zoom3d(recon, zoom_factor)
#    recon_adjoint[g] = recon_interp
## example realization
#ims_na_adjoint =[{'cmap':'viridis', 'vmax':true_na_image.max(), 'vmin':0.}] + [{'cmap':'viridis', 'vmax':im.max(), 'vmin':0.} for im in recon_adjoint_noiseless]
#ims = [true_na_image, recon_adjoint[0], recon_adjoint[1], recon_adjoint[2]]
#v.append(pv.ThreeAxisViewer(ims, imshow_kwargs=ims_na_adjoint, rowlabels=['','','',''], ls=''))
#plt.savefig(save_dir / f'brainweb_adjoint_recons_grads_annot.png', bbox_inches='tight')
## noiseless
#ims = [true_na_image, recon_adjoint_noiseless[0], recon_adjoint_noiseless[1], recon_adjoint_noiseless[2]]
#v.append(pv.ThreeAxisViewer(ims, imshow_kwargs=ims_na_adjoint, rowlabels=['True', 'Grad 0.16', 'Grad 0.32', 'Grad 0.48'], ls=''))
#plt.savefig(save_dir / f'brainweb_adjoint_noiseless_recons_grads_annot.png', bbox_inches='tight')

# ----------------------------------------------------------------
# init seaborn
sns.set_context('notebook')
sns.set(font_scale=1.5)
sns.set_style('ticks')

# -----------------------------------------------------------------------------
# Compute and display metrics

# keep only columns with quantitative metrics
# to be able to perform mean/std aggregate operations
df_stats = df.drop(columns='seed').copy()
# list of quantitative metrics to be further processed and analyzed
metrics = df_stats.columns.to_list()
for p in params:
    metrics.remove(p)
# the number of rows for each group should be = number of seeds/realizations
ddf = df_stats.groupby(params)
# compute the mean and stddev
df_stats = ddf.agg(['mean', 'std'])  # ddf
# flatten the index multiindex for use with FacetGrid
df_stats = df_stats.reset_index()
# flatten the columns multiindex for use with FacetGrid
df_stats.columns = [' '.join(col).strip() for col in df_stats.columns.values]

# compute bias/std in perc and snr for metrics when applicable
# doesn't make sense for image observer
# the snr of the contrast is the cnr
for i, c in enumerate(metrics):
    df_stats[c + ' snr'] = df_stats[c + ' mean']**2 / df_stats[c + ' std']**2
    if 'im_obs' not in c:
        df_stats[c + ' bias[%]'] = 100 * (
            df_stats[c + ' mean'].values - true[c]) / np.abs(true[c])
        df_stats[c + ' std[%]'] = 100 * df_stats[c + ' std'].values / np.abs(
            true[c])

#-------------------------------------------------------------------------------------
# Addition
# compute less noisy estimates of the CNR:
# compute the contrasts from noiseless images
# estimate the var using the sum of individual ROI vars
df_stats_orig = df_stats.copy()
for r, roi in enumerate(rois_for_contrast):
    for b, beta in enumerate(betas):
        for g, grad in enumerate(grads):
            roi_mean = recon_noiseless[g, b][masks[roi]].mean()
            ref_wm_roi_mean = recon_noiseless[g, b][masks['ref_wm']].mean()
            df_stats.loc[(df_stats['grad'] == grad) &
                         (df_stats['beta'] == beta), roi +
                         '_contrast_wm mean'] = float(roi_mean -
                                                      ref_wm_roi_mean)
            if load_baseline_realiz:
                roi_mean_baseline = recon_noiseless_baseline[g, b][
                    masks[roi]].mean()
                df_stats.loc[(df_stats['grad'] == grad) &
                             (df_stats['beta'] == beta), roi +
                             '_contrast_baseline mean'] = float(
                                 roi_mean - roi_mean_baseline)

    # compute an estimate of the std dev of the contrast
    var_sum = np.square(df_stats[roi + ' std'].values) + np.square(
        df_stats['ref_wm std'].values)
    df_stats[roi + '_contrast_wm std'] = np.sqrt(var_sum / 2.)

    # compute bias, snr
    metric = roi + '_contrast_wm'
    df_stats[
        metric +
        ' snr'] = df_stats[metric + ' mean']**2 / df_stats[metric + ' std']**2
    df_stats[metric + ' bias[%]'] = 100 * (df_stats[metric + ' mean'].values -
                                           true[metric]) / np.abs(true[metric])
    df_stats[metric + ' std[%]'] = 100 * (
        df_stats[metric + ' std'].values) / np.abs(true[metric])

    if load_baseline_realiz:
        # compute an estimate of the std dev of the contrast
        var_sum = np.square(df_stats[roi + ' std'].values) + np.square(
            df_stats[roi + '_baseline std'].values)
        df_stats[roi + '_contrast_baseline std'] = np.sqrt(var_sum / 2.)
        # compute snr and bias
        metric = roi + '_contrast_baseline'
        df_stats[metric + ' snr'] = df_stats[metric + ' mean']**2 / df_stats[
            metric + ' std']**2
        df_stats[metric + ' bias[%]'] = 100 * (
            df_stats[metric + ' mean'].values - true[metric]) / np.abs(
                true[metric])
        df_stats[metric + ' std[%]'] = 100 * (
            df_stats[metric + ' std'].values) / np.abs(true[metric])

# rename columns for display
# shorten some column names
df_stats.rename(columns=lambda x: re.sub('contrast', 'cntr', x), inplace=True)
df_stats.rename(columns=lambda x: re.sub('baseline', 'bsln', x), inplace=True)
metrics = [re.sub('contrast', 'cntr', x) for x in metrics]
metrics = [re.sub('baseline', 'bsln', x) for x in metrics]
for old, new in roi_rename_dict.items():
    metrics = [re.sub(old, new, x) for x in metrics]
    df_stats.rename(columns=lambda x: re.sub(old, new, x), inplace=True)

# display
for col in metrics:
    # show plots of snr/cnr for comparing gradients
    grid = sns.relplot(
        data=df_stats,
        kind='line',
        x=col + ' std',
        y=col + ' snr',
        hue="grad",
        marker='o',
        sort=False,
        markersize=5,
        legend='brief')
    for i, ax in enumerate(grid.axes.ravel()):
        ax.grid(ls=':')
        ax.plot([0.], [0.], markersize=15, marker='P', color='k')

        # display beta values for one gradient for checking
        df_temp = df_stats[df_stats['grad'] == 16]
        for r in range(df_temp.shape[0]):
            ax.annotate(
                r'$\beta$=' + f"{df_temp.at[r, 'beta']}",
                xy=(df_temp.at[r, col + ' std'], df_temp.at[r, col + ' snr']),
                fontsize='x-small')
    grid.fig.show()
    plt.savefig(save_dir / f"snr_std_{col}.pdf", bbox_inches='tight')

    # no bias for image observers
    if 'im_obs' in col:
        continue

    # show plots of snr vs bias for comparing gradients
    grid = sns.relplot(
        data=df_stats,
        kind='line',
        x=col + ' bias[%]',
        y=col + ' snr',
        hue="grad",
        marker='o',
        sort=False,
        markersize=5,
        legend='brief')
    for i, ax in enumerate(grid.axes.ravel()):
        ax.grid(ls=':')
        ax.plot([0.], [0.], markersize=15, marker='P', color='k')

        # display beta values for one gradient for checking
        df_temp = df_stats[df_stats['grad'] == 16]
        for r in range(df_temp.shape[0]):
            ax.annotate(
                r'$\beta$=' + f"{df_temp.at[r, 'beta']}",
                xy=(df_temp.at[r, col + ' bias[%]'],
                    df_temp.at[r, col + ' snr']),
                fontsize='x-small')
    grid.fig.show()
    plt.savefig(save_dir / f"snr_bias_{col}.pdf", bbox_inches='tight')

    # show plots of bias-stddev in perc for comparing gradients
    grid = sns.relplot(
        data=df_stats,
        kind="line",
        x=col + ' bias[%]',
        y=col + ' std[%]',
        hue="grad",
        marker='o',
        sort=False,
        markersize=5,
        legend='brief')

    for i, ax in enumerate(grid.axes.ravel()):
        ax.grid(ls=':')
        ax.plot([0.], [0.], markersize=15, marker='P', color='k')

        # display beta values for one gradient for checking
        df_temp = df_stats[df_stats['grad'] == 16]
        for r in range(df_temp.shape[0]):
            ax.annotate(
                r'$\beta$=' + f"{df_temp.at[r, 'beta']}",
                xy=(df_temp.at[r, col + ' bias[%]'],
                    df_temp.at[r, col + ' std[%]']),
                fontsize='x-small')
    grid.fig.show()
    plt.savefig(save_dir / f"bias_std_{col}.pdf", bbox_inches='tight')

# script duration
duration = time.time() - start_time
print(f'took {duration}s')

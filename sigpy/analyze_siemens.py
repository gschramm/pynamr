import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import find_objects, binary_dilation, binary_erosion
import SimpleITK as sitk
import pandas as pd

from collections import OrderedDict
from pathlib import Path
from utils import numpy_volume_to_sitk_image, sitk_image_to_numpy_volume

import argparse


def resample_sodium_to_t1_grid(na_image,
                               na_voxsize,
                               na_origin,
                               t1,
                               t1_voxsize,
                               t1_origin,
                               final_transform,
                               missing=0.0):
    fixed_sitk_image = numpy_volume_to_sitk_image(na_image.astype(np.float32),
                                                  na_voxsize, na_origin)
    moving_sitk_image = numpy_volume_to_sitk_image(
        np.flip(t1, (0, 2)).astype(np.float32), t1_voxsize, t1_origin)

    x_sitk = sitk.Resample(fixed_sitk_image, moving_sitk_image,
                           final_transform.GetInverse(), sitk.sitkLinear,
                           missing, fixed_sitk_image.GetPixelID())

    return np.flip(sitk_image_to_numpy_volume(x_sitk), (0, 2))


#-----------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--pnum', type=int, default=31)
args = parser.parse_args()

pnum = args.pnum

sdir = 'recons_128'
agr_w_decay_file = 'agr_both_echo_w_decay_model_L1_3.0E-03_3.0E-01_2000_19.npz'
agr_wo_decay_file = 'agr_echo_1_no_decay_model_L1_3.0E-03_2000.npz'
ratio_file = 'est_ratio_3.0E-03_3.0E-01_100_19.npy'
conv_file = 'recon_echo_1_no_decay_model_L2_1.0E-01_2000.npz'

pdir = Path(f'/data/sodium_mr/NYU/CSF{pnum}_raw')

# load the t1
t1_nii = nib.as_closest_canonical(nib.load(pdir / 't1.nii'))
t1 = t1_nii.get_fdata()

t1_affine = t1_nii.affine
t1_voxsize = t1_nii.header['pixdim'][1:4]
t1_origin = t1_affine[:-1, -1]

# load the freesurfer seg
aparc_nii = nib.as_closest_canonical(nib.load(pdir / 'aparc+aseg-native.nii'))
aparc = aparc_nii.get_fdata()

# load the transform that aligns the T1 to the sodium recons
na_origin = np.loadtxt(pdir / sdir / 'na_origin.txt')
final_transform = sitk.ReadTransform(str(pdir / sdir / 't1_transform.tfm'))

# load the AGR including decay model
na_voxsize = 10 * 22 / np.array([128, 128, 128])
na_origin = t1_origin.copy()

# load and resample the sodium recons
conv = resample_sodium_to_t1_grid(
    np.abs(np.load(pdir / sdir / conv_file)['x']), na_voxsize, na_origin, t1,
    t1_voxsize, t1_origin, final_transform)

agr_wo_decay = resample_sodium_to_t1_grid(
    np.abs(np.load(pdir / sdir / agr_wo_decay_file)['x']), na_voxsize,
    na_origin, t1, t1_voxsize, t1_origin, final_transform)

agr_w_decay = resample_sodium_to_t1_grid(
    np.abs(np.load(pdir / sdir / agr_w_decay_file)['x']), na_voxsize,
    na_origin, t1, t1_voxsize, t1_origin, final_transform)

est_ratio = resample_sodium_to_t1_grid(np.abs(np.load(pdir / sdir /
                                                      ratio_file)),
                                       na_voxsize,
                                       na_origin,
                                       t1,
                                       t1_voxsize,
                                       t1_origin,
                                       final_transform,
                                       missing=1.0)

#------------------------------------------------------------------------------
# flip all images to LPS

t1 = np.flip(t1, (0, 1))
conv = np.flip(conv, (0, 1))
agr_wo_decay = np.flip(agr_wo_decay, (0, 1))
agr_w_decay = np.flip(agr_w_decay, (0, 1))
est_ratio = np.flip(est_ratio, (0, 1))
aparc = np.flip(aparc, (0, 1))

#------------------------------------------------------------------------------
# flip define ROIs and quantify

agr_wo_means = OrderedDict()
agr_w_means = OrderedDict()
conv_means = OrderedDict()

roi_inds = OrderedDict()
cortical_gm_mask = (aparc >= 1000).astype(np.uint8)
wm_mask = (aparc == 2).astype(np.uint8) + (aparc == 41).astype(np.uint8)
# WM next to GM
wm_gm_mask = binary_dilation(cortical_gm_mask, iterations=2) * wm_mask
central_wm_mask = binary_erosion(wm_mask, iterations=5)
ventricle_mask = (aparc == 4).astype(np.uint8) + (aparc == 43).astype(np.uint8)

roi_inds['cortical GM'] = np.where(cortical_gm_mask)
roi_inds['WM'] = np.where(wm_mask)
roi_inds['WM next to GM'] = np.where(wm_gm_mask)
roi_inds['central WM'] = np.where(central_wm_mask)
roi_inds['ventricles'] = np.where(ventricle_mask)

for roi, inds in roi_inds.items():
    agr_wo_means[roi] = agr_wo_decay[inds].mean()
    agr_w_means[roi] = agr_w_decay[inds].mean()
    conv_means[roi] = conv[inds].mean()

df = pd.DataFrame({
    'conventional': pd.Series(conv_means),
    'AGR wo decay model': pd.Series(agr_wo_means),
    'AGR w decay model': pd.Series(agr_w_means)
})

df.loc['cortical GM / WM ratio'] = df.loc['cortical GM'] / df.loc['WM']
df.loc['cortical GM / WM next to GM ratio'] = df.loc['cortical GM'] / df.loc[
    'WM next to GM']
df.loc['cortical GM / central WM ratio'] = df.loc['cortical GM'] / df.loc[
    'central WM']

print(df)

#------------------------------------------------------------------------------

import pymirc.viewer as pv

bbox = find_objects(t1 > 0.1 * np.percentile(t1, 99.9))[0]

ims = [
    dict(cmap='Greys_r', vmin=0, vmax=np.percentile(t1, 99.99))
] + 3 * [dict(cmap='Greys_r', vmin=0, vmax=np.percentile(agr_w_decay, 99.9))
         ] + [dict(cmap='Greys_r', vmin=0, vmax=1)]

vi = pv.ThreeAxisViewer([
    t1[bbox], conv[bbox], agr_wo_decay[bbox], agr_w_decay[bbox],
    est_ratio[bbox]
],
                        imshow_kwargs=ims)

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import find_objects
import SimpleITK as sitk

from pathlib import Path
from utils import numpy_volume_to_sitk_image, sitk_image_to_numpy_volume


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

pdir = Path('/data/sodium_mr/NYU/CSF31_raw')
sdir = 'recons_128'
agr_w_decay_file = 'agr_both_echo_w_decay_model_L1_3.0E-03_3.0E-01_2000_19.npz'
agr_wo_decay_file = 'agr_echo_1_no_decay_model_L1_3.0E-03_2000.npz'
ratio_file = 'est_ratio_3.0E-03_3.0E-01_100_19.npy'
conv_file = 'recon_echo_1_no_decay_model_L2_1.0E-01_2000.npz'

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

import pymirc.viewer as pv

bbox = find_objects(t1 > 0.1 * np.percentile(t1, 99.9))[0]

ims = [
    dict(cmap='Greys_r', vmin=0, vmax=np.percentile(t1, 99.9))
] + 3 * [dict(cmap='Greys_r', vmin=0, vmax=np.percentile(agr_w_decay, 99.9))
         ] + [dict(cmap='Greys_r', vmin=0, vmax=1)]

vi = pv.ThreeAxisViewer([
    t1[bbox], conv[bbox], agr_wo_decay[bbox], agr_w_decay[bbox],
    est_ratio[bbox]
],
                        imshow_kwargs=ims)

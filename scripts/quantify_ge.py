import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pymirc.viewer as pv
from scipy.ndimage import binary_erosion

from pathlib import Path
from utils import numpy_volume_to_sitk_image, sitk_image_to_numpy_volume

field_of_view_cm = 22
ishape = (128, 128, 128)

odir = Path('/data/sodium_mr/20230225_MR3_GS_TPI/recons')

na_recon = np.abs(
    np.load(odir / 'agr_both_echo_w_decay_model_L1_1.0E-02.npz')['x'])
#    np.load(odir / 'recon_echo_1_no_decay_model_L2_3.0E-02.npz')['x'])
na_origin = np.loadtxt(odir / 'na_origin.txt')
na_voxsize = 10 * field_of_view_cm / np.array(ishape)

# load aparc
aparc_nii = nib.load(odir.parent / 'fastsurfer' /
                     'aparc.DKTatlas+aseg.deep.mgz')
aparc_nii = nib.as_closest_canonical(aparc_nii)
aparc = aparc_nii.get_fdata()

aparc_affine = aparc_nii.affine
aparc_voxsize = aparc_nii.header['delta']
aparc_origin = aparc_affine[:-1, -1]

tfm = sitk.ReadTransform(str(odir / 't1_transform.tfm'))

#------------------------------------------------------------

na_sitk = numpy_volume_to_sitk_image(na_recon, na_voxsize, na_origin)
aparc_sitk = numpy_volume_to_sitk_image(aparc, aparc_voxsize, aparc_origin)

# transform of sodium recon grid to T1 grid

na_sitk_resampled = sitk.Resample(na_sitk, aparc_sitk, tfm.GetInverse(),
                                  sitk.sitkLinear, 0.0, na_sitk.GetPixelID())

na_resampled = sitk_image_to_numpy_volume(na_sitk_resampled)

gm_mask = binary_erosion(aparc >= 1000)
wm_mask = binary_erosion((aparc == 2) + (aparc == 41), iterations=3)
csf_mask = binary_erosion((aparc == 4) + (aparc == 43), iterations=3)

print(na_resampled[gm_mask].mean())
print(na_resampled[wm_mask].mean())
print(na_resampled[csf_mask].mean())

vi = pv.ThreeAxisViewer([na_resampled, na_resampled, na_resampled],
                        [gm_mask, wm_mask, csf_mask],
                        imshow_kwargs=dict(cmap='Greys_r', vmin=0, vmax=3))

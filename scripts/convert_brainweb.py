import SimpleITK as sitk
import nibabel as nib
import numpy as np
import pymirc.viewer as pv

from utils import sitk_image_to_numpy_volume

from pathlib import Path


def convert_to_nifti(fname: Path) -> Path:
    nib_image = nib.load(str(fname))
    oname = fname.with_suffix('.nii')
    nib.save(
        nib.Nifti1Image(np.asanyarray(nib_image.dataobj), nib_image.affine),
        oname)

    return oname


t1_path = Path('/data/sodium_mr/brainweb54/subject54_t1w_p4.mnc')
t1_nii_path = convert_to_nifti(t1_path)

seg_path = Path('/data/sodium_mr/brainweb54/subject54_crisp_v.mnc')
seg_nii_path = convert_to_nifti(seg_path)

moving_sitk_image = sitk.ReadImage(t1_nii_path)
fixed_sitk_image = sitk.ReadImage(seg_nii_path)

moving_sitk_image_resampled = sitk.Resample(moving_sitk_image,
                                            fixed_sitk_image,
                                            sitk.Euler3DTransform(),
                                            sitk.sitkLinear, 0.0,
                                            moving_sitk_image.GetPixelID())

sitk.WriteImage(moving_sitk_image_resampled,
                t1_nii_path.parent / f'{t1_nii_path.stem}_resampled.nii')

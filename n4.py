import SimpleITK as sitk
from pathlib import Path

fname = Path('data/sodium_data/BT12V6/mprage.nii')

inputImage = sitk.ReadImage(str(fname), sitk.sitkFloat32)
image      = inputImage

maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)

corrector = sitk.N4BiasFieldCorrectionImageFilter()

numberFittingLevels = 4

corrected_image = corrector.Execute(image, maskImage)

sitk.WriteImage(corrected_image, str(fname.parent / 'T1_n4.nii'))

import numpy as np
a = np.flip(np.swapaxes(sitk.GetArrayFromImage(inputImage),0,2), (0,1))
b = np.flip(np.swapaxes(sitk.GetArrayFromImage(corrected_image),0,2), (0,1))

import pymirc.viewer as pv
import matplotlib.pyplot as plt
pv.ThreeAxisViewer([a,b], imshow_kwargs = {'cmap':plt.cm.Greys_r})

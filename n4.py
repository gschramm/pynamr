import SimpleITK as sitk
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('case')
parser.add_argument('--sdir', default = 'mprage_proton')
parser.add_argument('--fname', default = 'T1.nii')
args = parser.parse_args()

fname = Path('data/sodium_data') / args.case / args.sdir / args.fname

#-------------------------------------------

inputImage = sitk.ReadImage(str(fname), sitk.sitkFloat32)
image      = inputImage

maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)

corrector = sitk.N4BiasFieldCorrectionImageFilter()

numberFittingLevels = 4

corrected_image = corrector.Execute(image, maskImage)

sitk.WriteImage(corrected_image, str(fname.parent / '_n4.'.join(fname.name.split('.'))))

import numpy as np
a = np.flip(np.swapaxes(sitk.GetArrayFromImage(inputImage),0,2), (0,1))
b = np.flip(np.swapaxes(sitk.GetArrayFromImage(corrected_image),0,2), (0,1))

import pymirc.viewer as pv
import matplotlib.pyplot as plt
vi = pv.ThreeAxisViewer([a,b], imshow_kwargs = {'cmap':plt.cm.Greys_r})

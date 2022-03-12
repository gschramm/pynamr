from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import SimpleITK as sitk
import pymirc.viewer as pv
import matplotlib.pyplot as plt

def numpy_volume_to_sitk_image(vol, voxel_size, origin):
  image = sitk.GetImageFromArray(np.swapaxes(vol, 0, 2))
  image.SetSpacing(voxel_size.astype(np.float64))
  image.SetOrigin(origin.astype(np.float64))

  return image

def sitk_image_to_numpy_volume(image):
  vol = np.swapaxes(sitk.GetArrayFromImage(image), 0, 2)

  return vol

#----------------------------------------------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument('case')
parser.add_argument('--sdir', default = 'DeNoise_kw0')
parser.add_argument('--n', default = 128, type = int)
args = parser.parse_args()

case = args.case
pdir = Path('data') / 'sodium_data' / args.case
sdir = args.sdir
n    = args.n

data_shape  = (64,64,64)
recon_shape = (n,n,n) 

#----------------------------------------------------------------------------------------------------
sos_file = list(pdir.glob(f'*TE03*/{sdir.split("_")[0]}/{sdir}/*.csos'))[0]

# load the N4 corrected T1 in RAS
t1_nii     = nib.load(pdir / 'mprage_proton' / 'T1_n4.nii')
t1_nii     = nib.as_closest_canonical(t1_nii)
t1         = t1_nii.get_fdata()
t1_affine  = t1_nii.affine
t1_voxsize = t1_nii.header['pixdim'][1:4]
t1_origin  = t1_affine[:-1,-1]

# the sum of squares Na image
sos = np.flip(np.fromfile(sos_file, dtype = np.complex64).reshape(data_shape).swapaxes(0,2), (1,2)).real

# interpolate to smaller grid
sos = zoom(sos, np.array(recon_shape)/ np.array(data_shape), order = 1, prefilter = False)

# construct the voxel size for the Na sos image
fov = 220.
sos_voxsize = fov / np.array(recon_shape)
sos_origin = np.full(3,-fov/2.)

#----------------------------------------------------------------------------------------------------

fixed_image  = numpy_volume_to_sitk_image(sos.astype(np.float32), sos_voxsize, sos_origin)
moving_image = numpy_volume_to_sitk_image(t1.astype(np.float32),  t1_voxsize,  t1_origin)

# Initial Alignment
initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image,
                                                      sitk.Euler3DTransform(),
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)

# Registration
registration_method = sitk.ImageRegistrationMethod()

# Similarity metric settings.
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins = 50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.1)

registration_method.SetInterpolator(sitk.sitkLinear)

# Optimizer settings.
registration_method.SetOptimizerAsGradientDescentLineSearch(
    learningRate            = 0.3,
    numberOfIterations      = 200,
    convergenceMinimumValue = 1e-6,
    convergenceWindowSize   = 10)

registration_method.SetOptimizerScalesFromPhysicalShift()

# Setup for the multi-resolution framework.
registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4, 2, 1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2, 1, 0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# Don't optimize in-place, we would possibly like to run this cell multiple times.
registration_method.SetInitialTransform(initial_transform, inPlace = False)

final_transform = registration_method.Execute(
    sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32))

# Post registration analysis
print(f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}")
print(f"Final metric value: {registration_method.GetMetricValue()}")
print(f"Final parameters: {final_transform.GetParameters()}")

moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                 moving_image.GetPixelID())


t1_aligned = sitk_image_to_numpy_volume(moving_resampled)

# save aligned T1
np.save(pdir / 'mprage_proton' / f'T1_n4_aligned_{recon_shape[0]}.nii', t1_aligned)
#----------------------------------------------------------------------------------------------------

vi = pv.ThreeAxisViewer([t1_aligned,sos,t1_aligned],[None,None,sos], imshow_kwargs = {'cmap':plt.cm.Greys_r})

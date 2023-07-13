from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from typing import Sequence, Union


def numpy_volume_to_sitk_image(vol, voxel_size, origin):
    image = sitk.GetImageFromArray(np.swapaxes(vol, 0, 2))
    image.SetSpacing(voxel_size.astype(np.float64))
    image.SetOrigin(origin.astype(np.float64))

    return image


def sitk_image_to_numpy_volume(image):
    vol = np.swapaxes(sitk.GetArrayFromImage(image), 0, 2)

    return vol


def align_images(fixed_image: np.ndarray,
                 moving_image: np.ndarray,
                 fixed_voxsize: Sequence[float] = (1., 1., 1.),
                 fixed_origin: Sequence[float] = (0., 0., 0.),
                 moving_voxsize: Sequence[float] = (1., 1., 1.),
                 moving_origin: Sequence[float] = (0., 0., 0.),
                 final_transform: sitk.Transform | None = None,
                 verbose: bool = True):

    fixed_sitk_image = numpy_volume_to_sitk_image(
        fixed_image.astype(np.float32), fixed_voxsize, fixed_origin)
    moving_sitk_image = numpy_volume_to_sitk_image(
        moving_image.astype(np.float32), moving_voxsize, moving_origin)

    if final_transform is None:
        # Initial Alignment
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_sitk_image, moving_sitk_image, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY)

        # Registration
        registration_method = sitk.ImageRegistrationMethod()

        # Similarity metric settings.
        registration_method.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(
            registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.1)

        registration_method.SetInterpolator(sitk.sitkLinear)

        # Optimizer settings.
        registration_method.SetOptimizerAsGradientDescentLineSearch(
            learningRate=0.2,
            numberOfIterations=400,
            convergenceMinimumValue=1e-7,
            convergenceWindowSize=10)

        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Setup for the multi-resolution framework.
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(
            smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Don't optimize in-place, we would possibly like to run this cell multiple times.
        registration_method.SetInitialTransform(initial_transform,
                                                inPlace=False)

        final_transform = registration_method.Execute(
            sitk.Cast(fixed_sitk_image, sitk.sitkFloat32),
            sitk.Cast(moving_sitk_image, sitk.sitkFloat32))

        # Post registration analysis
        if verbose:
            print(
                f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}"
            )
            print(
                f"Final metric value: {registration_method.GetMetricValue()}")
            print(f"Final parameters: {final_transform.GetParameters()}")

    moving_sitk_image_resampled = sitk.Resample(moving_sitk_image,
                                                fixed_sitk_image,
                                                final_transform,
                                                sitk.sitkLinear, 0.0,
                                                moving_sitk_image.GetPixelID())

    moving_image_aligned = sitk_image_to_numpy_volume(
        moving_sitk_image_resampled)

    return moving_image_aligned, final_transform

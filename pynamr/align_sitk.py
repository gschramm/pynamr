import SimpleITK as sitk


def resample_volume(volume: sitk.Image,
                    new_spacing: tuple[float, float, float],
                    interpolator: int = sitk.sitkLinear) -> sitk.Image:
    """ resample volume to differnt voxel size"""
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [
        int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(
            original_size, original_spacing, new_spacing)
    ]
    return sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                         volume.GetOrigin(), new_spacing,
                         volume.GetDirection(), 0, volume.GetPixelID())


def align_sitk_images(fixed_image: sitk.Image,
                      moving_image: sitk.Image,
                      new_spacing: tuple[float, float, float] = (1., 1., 1.),
                      sampling_rate: float = 0.01,
                      initial_transform: sitk.Transform = None,
                      registration_method: sitk.ImageRegistrationMethod = None,
                      verbose: bool = False) -> tuple[sitk.Image, sitk.Image]:
    """ align two SITK images and interpolate to a nomimal voxel spacing if needed """

    if not tuple(new_spacing) == fixed_image.GetSpacing():
        fixed_image = resample_volume(fixed_image, new_spacing)

    # Initial transform -> align image centers based on affine
    if initial_transform is None:
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY)

    # Registration
    if registration_method is None:
        registration_method = sitk.ImageRegistrationMethod()

        # Similarity metric settings.
        registration_method.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(
            registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(sampling_rate)

        registration_method.SetInterpolator(sitk.sitkLinear)

        # Optimizer settings.
        registration_method.SetOptimizerAsGradientDescentLineSearch(
            learningRate=1.0,
            numberOfIterations=100,
            convergenceMinimumValue=1e-6,
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

    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Post registration analysis
    if verbose:
        print(
            f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}"
        )
        print(f"Final metric value: {registration_method.GetMetricValue()}")
        print(f"Final parameters: {final_transform.GetParameters()}")

    moving_image_aligned = sitk.Resample(moving_image, fixed_image,
                                         final_transform, sitk.sitkLinear, 0.0,
                                         moving_image.GetPixelID())

    return fixed_image, moving_image_aligned


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='align two 3D images using simpleITK')
    parser.add_argument('fixed_file',
                        type=str,
                        help='file name of fixed image')
    parser.add_argument('moving_file',
                        type=str,
                        help='file name of moving image')
    parser.add_argument('--new_spacing',
                        type=float,
                        nargs='+',
                        default=[1., 1., 1.],
                        help='file name of moving image')
    parser.add_argument('--show', help='show results', action='store_true')
    parser.add_argument('--verbose',
                        help='verbose output',
                        action='store_true')
    args = parser.parse_args()

    new_spacing = tuple(args.new_spacing)

    f_image = sitk.ReadImage(args.fixed_file, sitk.sitkFloat32)
    m_image = sitk.ReadImage(args.moving_file, sitk.sitkFloat32)

    f_image_interp, m_image_aligned = align_sitk_images(
        f_image, m_image, new_spacing=new_spacing, verbose=args.verbose)

    # write the output images
    f_path = Path(args.fixed_file)
    fo_path = str(f_path.parent / f'{f_path.stem}_interp{f_path.suffix}')
    sitk.WriteImage(f_image_interp, fo_path)

    m_path = Path(args.moving_file)
    mo_path = str(m_path.parent / f'{m_path.stem}_aligned{m_path.suffix}')
    sitk.WriteImage(m_image_aligned, mo_path)

    if args.verbose:
        print(fo_path)
        print(mo_path)

    if args.show:
        import pymirc.viewer as pv
        import matplotlib.pyplot as plt
        vi = pv.ThreeAxisViewer([
            sitk.GetArrayViewFromImage(f_image_interp),
            sitk.GetArrayViewFromImage(m_image_aligned),
            sitk.GetArrayViewFromImage(f_image_interp)
        ], [None, None,
            sitk.GetArrayViewFromImage(m_image_aligned)],
                                imshow_kwargs={'cmap': plt.cm.Greys_r})

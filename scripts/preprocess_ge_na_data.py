from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pymirc.viewer as pv
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
from typing import Sequence
import argparse


def numpy_volume_to_sitk_image(vol, voxel_size, origin):
    image = sitk.GetImageFromArray(np.swapaxes(vol, 0, 2))
    image.SetSpacing(voxel_size.astype(np.float64))
    image.SetOrigin(origin.astype(np.float64))

    return image


def sitk_image_to_numpy_volume(image):
    vol = np.swapaxes(sitk.GetArrayFromImage(image), 0, 2)

    return vol


def window3D(w):
    # Convert a 1D filtering kernel to 3D
    # eg, window3D(numpy.hanning(5))
    L = w.shape[0]
    m1 = np.outer(np.ravel(w), np.ravel(w))
    win1 = np.tile(m1, np.hstack([L, 1, 1]))
    m2 = np.outer(np.ravel(w), np.ones([1, L]))
    win2 = np.tile(m2, np.hstack([L, 1, 1]))
    win2 = np.transpose(win2, np.hstack([1, 2, 0]))
    win = np.multiply(win1, win2)
    return win


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


#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--na_echo_1_file',
                        type=str,
                        required=True,
                        help='name of 1st echo sodium IFFT image')
    parser.add_argument('--na_echo_2_file',
                        type=str,
                        required=True,
                        help='name of 2nd echo sodium IFFT image')
    parser.add_argument('--t1_nifti_file',
                        type=str,
                        required=True,
                        help='name of T1 nifti')

    args = parser.parse_args()

    # input parameters
    na_echo_1_file: str = args.na_echo_1_file
    na_echo_2_file: str = args.na_echo_2_file
    t1_nifti_file: str = args.t1_nifti_file

    output_dir = Path(
        na_echo_1_file).parents[1] / 'preprocessed_regridded_data'
    output_dir.mkdir(exist_ok=True, parents=True)

    #----------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------

    # load the N4 corrected T1 in RAS
    t1_nii = nib.load(t1_nifti_file)
    t1_nii = nib.as_closest_canonical(t1_nii)
    t1 = t1_nii.get_fdata()
    t1_affine = t1_nii.affine
    t1_voxsize = t1_nii.header['pixdim'][1:4]
    t1_origin = t1_affine[:-1, -1]

    # the sum of squares Na image
    na_echo_1 = np.fromfile(na_echo_1_file, dtype=np.complex64)
    na_echo_2 = np.fromfile(na_echo_2_file, dtype=np.complex64)

    # reshape the Na data to a 3D image
    matrix_size = round(na_echo_1.size**(1 / 3))
    recon_shape = (matrix_size, matrix_size, matrix_size)
    na_echo_1 = na_echo_1.reshape(recon_shape).swapaxes(0, 2)
    na_echo_2 = na_echo_2.reshape(recon_shape).swapaxes(0, 2)

    # construct the voxel size for the Na sos image
    fov = 220.
    na_voxsize = fov / np.array(recon_shape)
    na_origin = np.full(3, -fov / 2.)

    # filter the na image with a hann filter in kspace
    tmp_x = np.arange(recon_shape[0])
    TMP_X, TMP_Y, TMP_Z = np.meshgrid(tmp_x, tmp_x, tmp_x)
    phase_correction = ((-1)**TMP_X) * ((-1)**TMP_Y) * ((-1)**TMP_Z)
    hann_win = np.fft.fftshift(window3D(hann(recon_shape[0]))).astype(
        np.float32)

    # calculate the kspace data
    echo_1_kspace_data = np.fft.fftn(na_echo_1 * phase_correction,
                                     norm='ortho').astype(np.complex64)
    echo_2_kspace_data = np.fft.fftn(na_echo_2 * phase_correction,
                                     norm='ortho').astype(np.complex64)

    # calculated smoothed images for registration
    na_echo_1_sm = np.abs(
        np.fft.ifftn(echo_1_kspace_data * hann_win, norm='ortho'))
    na_echo_2_sm = np.abs(
        np.fft.ifftn(echo_2_kspace_data * hann_win, norm='ortho'))

    # normalize the images and the data such that CSF in the first echo is ca. 3
    norm = np.percentile(na_echo_1_sm, 99.99) / 3

    echo_1_kspace_data /= norm
    echo_2_kspace_data /= norm

    na_echo_1_sm /= norm
    na_echo_2_sm /= norm

    na_echo_1 /= norm
    na_echo_2 /= norm

    #----------------------------------------------------------------------------------------------------

    t1_aligned, final_transform = align_images(na_echo_1_sm,
                                               t1,
                                               fixed_voxsize=na_voxsize,
                                               moving_voxsize=t1_voxsize,
                                               fixed_origin=na_origin,
                                               moving_origin=t1_origin)

    # save the preprocessed data
    print(f'writing data to {str(output_dir)}')
    np.save(output_dir / f'echo_1.npy', echo_1_kspace_data)
    np.save(output_dir / f'echo_2.npy', echo_2_kspace_data)
    np.save(output_dir / f't1_aligned.npy', t1_aligned)
    sitk.WriteTransform(final_transform,
                        str(output_dir / 't1_to_na_echo_1_transform.tfm'))

    #----------------------------------------------------------------------------------------------------

    vi = pv.ThreeAxisViewer([t1_aligned, na_echo_1_sm, t1_aligned],
                            [None, None, na_echo_1_sm],
                            imshow_kwargs={'cmap': plt.cm.Greys_r})
    vi.fig.savefig(output_dir / 't1_na_alignment.png')

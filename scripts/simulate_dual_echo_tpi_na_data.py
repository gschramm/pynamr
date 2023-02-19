"""script to regrid 3D TPI (twisted projection) k-space data"""

import argparse
import numpy as np
import math
import numpy.typing as npt
import nibabel as nib
from numba import jit
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_dilation

import pynufft

import pymirc.viewer as pv
from pymirc.image_operations import aff_transform, zoom3d


@jit(nopython=True)
def regrid_tpi_data(matrix_size: int,
                    delta_k: float,
                    data: npt.NDArray,
                    Nmax: int,
                    kx: npt.NDArray,
                    ky: npt.NDArray,
                    kz: npt.NDArray,
                    kmax: float,
                    kp: float,
                    window: npt.NDArray,
                    cutoff: float,
                    output: npt.NDArray,
                    correct_tpi_sampling_density=True,
                    output_weights=False) -> None:
    """ function to regrid 3D TPI (twisted projection) k-space data on regular k-space grid
        using tri-linear interpolation, correction for sampling density and windowing

    Parameters
    ----------
    matrix_size : int
        size of the k-space grid (matrix)
    delta_k : float
        the k-space spacing
    data : npt.NDArray
        1D numpy array with the complex k-space data values
    Nmax : int
        process only the first Nmax data points
    kx : npt.NDArray
        1D numpy array containing the x-component of the k-space coordinates
        unit 1/cm
    ky : npt.NDArray
        1D numpy array containing the y-component of the k-space coordinates
        unit 1/cm
    kz : npt.NDArray
        1D numpy array containing the z-component of the k-space coordinates
        unit 1/cm
    kmax : float
        kmax value
        unit 1/cm
    kp : float
        Euclidean distance in k-space until sampling is "radial" with sampling
        density 1/k**2
        unit 1/cm
    window : npt.NDArray
        1D array containing the window
        The window is indexed at: int((|k| / kmax) * cutoff)
    cutoff : float
        Cutoff parameter for window index
    output : npt.NDArray
        3D complex numpy array for the output
        if output_weights is True, it can be real
    correct_tpi_sampling_density: bool
        correct for the TPI sampling density (~k^2 if k <= kp)
    output_weights: bool
        output the regridding weights instead of the regridded data
    """

    for i in range(Nmax):
        # Euclidean distance from center of kspace points
        abs_k = np.sqrt(kx[i]**2 + ky[i]**2 + kz[i]**2)

        if abs_k <= kmax:
            # calculate the sampling density which is prop. to abs_k**2
            # in the inner part where sampling is radial
            # and constant in the outer part where the trajectories start
            # twisting

            if correct_tpi_sampling_density:
                if abs_k < kp:
                    sampling_density = abs_k**2
                else:
                    sampling_density = kp**2
            else:
                sampling_density = 1.

            # calculate the window index and value
            i_window = int((abs_k / kmax) * cutoff)

            # shift the kspace coordinates by half the matrix size to get the
            # origin k = (0,0,0) in the center of the array
            # we also divide by delta k, such that we get k in k_space grid units

            kx_shifted = (kx[i] / delta_k) + 0.5 * (matrix_size)
            ky_shifted = (ky[i] / delta_k) + 0.5 * (matrix_size)
            kz_shifted = (kz[i] / delta_k) + 0.5 * (matrix_size)

            # calculate the distances between the lower / upper cells
            # in the kspace grid
            kx_shifted_low = math.floor(kx_shifted)
            ky_shifted_low = math.floor(ky_shifted)
            kz_shifted_low = math.floor(kz_shifted)

            kx_shifted_high = kx_shifted_low + 1
            ky_shifted_high = ky_shifted_low + 1
            kz_shifted_high = kz_shifted_low + 1

            dkx = kx_shifted - kx_shifted_low
            dky = ky_shifted - ky_shifted_low
            dkz = kz_shifted - kz_shifted_low

            if output_weights:
                windowed_data = 1. + 0j
            else:
                windowed_data = data[i] * window[i_window]

            if (kx_shifted_low >= 0) and (ky_shifted_low >=
                                          0) and (kz_shifted_low >= 0):
                # fill output array according to trilinear interpolation
                output[kx_shifted_low, ky_shifted_low,
                       kz_shifted_low] += (1 - dkx) * (1 - dky) * (
                           1 - dkz) * sampling_density * windowed_data
                output[kx_shifted_high, ky_shifted_low,
                       kz_shifted_low] += (dkx) * (1 - dky) * (
                           1 - dkz) * sampling_density * windowed_data
                output[kx_shifted_low, ky_shifted_high,
                       kz_shifted_low] += (1 - dkx) * (dky) * (
                           1 - dkz) * sampling_density * windowed_data
                output[kx_shifted_high, ky_shifted_high, kz_shifted_low] += (
                    dkx) * (dky) * (1 - dkz) * sampling_density * windowed_data

                output[kx_shifted_low, ky_shifted_low,
                       kz_shifted_high] += (1 - dkx) * (1 - dky) * (
                           dkz) * sampling_density * windowed_data
                output[kx_shifted_high, ky_shifted_low, kz_shifted_high] += (
                    dkx) * (1 - dky) * (dkz) * sampling_density * windowed_data
                output[kx_shifted_low, ky_shifted_high, kz_shifted_high] += (
                    1 - dkx) * (dky) * (dkz) * sampling_density * windowed_data
                output[kx_shifted_high, ky_shifted_high, kz_shifted_high] += (
                    dkx) * (dky) * (dkz) * sampling_density * windowed_data


def read_single_tpi_gradient_file(gradient_file: str,
                                  gamma_by_2pi: float = 1126.2,
                                  num_header_elements: int = 6):

    header = np.fromfile(gradient_file,
                         dtype=np.int16,
                         offset=0,
                         count=num_header_elements)

    # number of cones
    num_cones = int(header[0])
    # number of points in a single readout
    num_points = int(header[1])

    # time sampling step in seconds
    dt = float(header[2]) * (1e-6)

    # maximum gradient strength in G/cm corresponds to max short value (2**15 - 1 = 32767
    max_gradient = float(header[3]) / 100

    # number of readouts per cone
    num_readouts_per_cone = np.fromfile(gradient_file,
                                        dtype=np.int16,
                                        offset=num_header_elements * 2,
                                        count=num_cones)

    gradient_array = np.fromfile(gradient_file,
                                 dtype=np.int16,
                                 offset=(num_header_elements + num_cones) * 2,
                                 count=num_cones * num_points).reshape(
                                     num_cones, num_points)

    # calculate k_array in (1/cm)
    k_array = np.cumsum(
        gradient_array,
        axis=1) * dt * gamma_by_2pi * max_gradient / (2**15 - 1)

    return k_array, header, num_readouts_per_cone


def read_tpi_gradient_files(file_base: str,
                            x_suffix: str = 'x.grdb',
                            y_suffix: str = 'y.grdb',
                            z_suffix: str = 'z.grdb',
                            **kwargs):

    kx, header, num_readouts_per_cone = read_single_tpi_gradient_file(
        f'{file_base}.{x_suffix}', **kwargs)
    ky, header, num_readouts_per_cone = read_single_tpi_gradient_file(
        f'{file_base}.{y_suffix}', **kwargs)
    kz, header, num_readouts_per_cone = read_single_tpi_gradient_file(
        f'{file_base}.{z_suffix}', **kwargs)

    kx_rotated = np.zeros((num_readouts_per_cone.sum(), kx.shape[1]))
    ky_rotated = np.zeros((num_readouts_per_cone.sum(), ky.shape[1]))
    kz_rotated = np.zeros((num_readouts_per_cone.sum(), kz.shape[1]))

    num_readouts_cumsum = np.cumsum(
        np.concatenate(([0], num_readouts_per_cone)))

    # start angle of first readout in each cone
    phi0s = np.linspace(0, 2 * np.pi, kx.shape[0], endpoint=False)

    for i_cone in range(header[0]):
        num_readouts = num_readouts_per_cone[i_cone]

        phis = np.linspace(phi0s[i_cone],
                           2 * np.pi + phi0s[i_cone],
                           num_readouts,
                           endpoint=False)

        for ir in range(num_readouts):
            kx_rotated[ir + num_readouts_cumsum[i_cone], :] = np.cos(
                phis[ir]) * kx[i_cone, :] - np.sin(phis[ir]) * ky[i_cone, :]
            ky_rotated[ir + num_readouts_cumsum[i_cone], :] = np.sin(
                phis[ir]) * kx[i_cone, :] + np.cos(phis[ir]) * ky[i_cone, :]
            kz_rotated[ir + num_readouts_cumsum[i_cone], :] = kz[i_cone, :]

    return kx_rotated, ky_rotated, kz_rotated, header, num_readouts_per_cone


def show_tpi_readout(kx,
                     ky,
                     kz,
                     header,
                     num_readouts_per_cone,
                     start_cone=0,
                     end_cone=None,
                     cone_step=2,
                     readout_step=6,
                     step=20):
    num_cones = header[0]

    if end_cone is None:
        end_cone = num_cones

    cone_numbers = np.arange(start_cone, end_cone, cone_step)

    # cumulative sum of readouts per cone
    rpc_cumsum = np.concatenate(([0], num_readouts_per_cone.cumsum()))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    for ic in cone_numbers:
        ax.scatter3D(
            kx[rpc_cumsum[ic]:rpc_cumsum[ic + 1]:readout_step, ::step],
            ky[rpc_cumsum[ic]:rpc_cumsum[ic + 1]:readout_step, ::step],
            kz[rpc_cumsum[ic]:rpc_cumsum[ic + 1]:readout_step, ::step],
            s=0.5)

    ax.set_xlim(kx.min(), kx.max())
    ax.set_ylim(ky.min(), ky.max())
    ax.set_zlim(kz.min(), kz.max())
    fig.tight_layout()
    fig.show()


#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------


def setup_brainweb_phantom(simulation_matrix_size: int,
                           phantom_data_path: Path,
                           field_of_view_cm: float = 22.,
                           csf_na_concentration: float = 3.0,
                           gm_na_concentration: float = 1.5,
                           wm_na_concentration: float = 1.0,
                           T2long_ms_csf: float = 50.,
                           T2long_ms_gm: float = 15.,
                           T2long_ms_wm: float = 18.,
                           T2short_ms_csf: float = 50.,
                           T2short_ms_gm: float = 8.,
                           T2short_ms_wm: float = 9.):

    simulation_voxel_size_mm: float = 10 * field_of_view_cm / simulation_matrix_size

    # setup the phantom on a high resolution grid (0.5^3mm) first
    label_nii = nib.load(phantom_data_path / 'subject54_crisp_v.mnc.gz')
    label_nii = nib.as_closest_canonical(label_nii)

    # pad o 434x434x434 voxels
    lab = np.pad(label_nii.get_fdata(), ((36, 36), (0, 0), (36, 36)),
                 'constant')

    lab_affine = label_nii.affine.copy()
    lab_affine[0, -1] -= 36 * lab_affine[0, 0]
    lab_affine[2, -1] -= 36 * lab_affine[2, 2]

    # CSF = 1, GM = 2, WM = 3
    csf_inds = np.where(lab == 1)
    gm_inds = np.where(lab == 2)
    wm_inds = np.where(lab == 3)

    # set up array for trans. magnetization
    img = np.zeros(lab.shape, dtype=np.float32)
    img[csf_inds] = csf_na_concentration
    img[gm_inds] = gm_na_concentration
    img[wm_inds] = wm_na_concentration

    # set up array for Gamma (ratio between 2nd and 1st echo)
    T2short_ms = np.full(lab.shape,
                         0.5 * np.finfo(np.float32).max,
                         dtype=np.float32)
    T2short_ms[csf_inds] = T2short_ms_csf
    T2short_ms[gm_inds] = T2short_ms_gm
    T2short_ms[wm_inds] = T2short_ms_wm

    T2long_ms = np.full(lab.shape,
                        0.5 * np.finfo(np.float32).max,
                        dtype=np.float32)
    T2long_ms[csf_inds] = T2long_ms_csf
    T2long_ms[gm_inds] = T2long_ms_gm
    T2long_ms[wm_inds] = T2long_ms_wm

    # read the T1 and interpolate to the grid of the high-res image
    t1_nii = nib.load(phantom_data_path / 'subject54_t1w_p4.mnc.gz')
    t1_nii = nib.as_closest_canonical(t1_nii)
    t1 = t1_nii.get_fdata()
    t1 = aff_transform(t1,
                       np.linalg.inv(t1_nii.affine) @ lab_affine,
                       lab.shape,
                       cval=t1.min())

    # extrapolate the all images to the voxel size we need for the data simulation
    img_extrapolated = zoom3d(img, lab_affine[0, 0] / simulation_voxel_size_mm)
    T2short_ms_extrapolated = zoom3d(
        T2short_ms, lab_affine[0, 0] / simulation_voxel_size_mm)
    T2long_ms_extrapolated = zoom3d(
        T2long_ms, lab_affine[0, 0] / simulation_voxel_size_mm)
    t1_extrapolated = zoom3d(t1, lab_affine[0, 0] / simulation_voxel_size_mm)

    # since the FOV of the brainweb label image is slightly smaller than 220mm, we
    # have to pad the image

    pad_width = simulation_matrix_size - img_extrapolated.shape[0]
    p0 = pad_width // 2
    p1 = pad_width - p0

    img_extrapolated = np.pad(img_extrapolated, (p0, p1))
    T2short_ms_extrapolated = np.pad(
        T2short_ms_extrapolated, (p0, p1),
        constant_values=T2short_ms_extrapolated.max())
    T2long_ms_extrapolated = np.pad(
        T2long_ms_extrapolated, (p0, p1),
        constant_values=T2long_ms_extrapolated.max())
    t1_extrapolated = np.pad(t1_extrapolated, (p0, p1))

    ## export GM and WM image
    #from scipy.ndimage import zoom

    #aparc_nii = nib.load(phantom_data_path / 'aparc+aseg_native.nii.gz')
    #aparc_nii = nib.as_closest_canonical(aparc_nii)

    #cortex = (aparc_nii.get_fdata() >= 1000)

    #cortex = aff_transform(cortex,
    #                       np.linalg.inv(aparc_nii.affine) @ lab_affine,
    #                       lab.shape,
    #                       cval=t1.min())

    #gm = (lab == 2)
    #wm = (lab == 3)

    #gm_extrapolated = np.pad(
    #    zoom(gm, lab_affine[0, 0] / simulation_voxel_size_mm, order=0),
    #    (p0, p1))
    #wm_extrapolated = np.pad(
    #    zoom(wm, lab_affine[0, 0] / simulation_voxel_size_mm, order=0),
    #    (p0, p1))
    #cortex_extrapolated = np.pad(
    #    zoom(cortex, lab_affine[0, 0] / simulation_voxel_size_mm, order=0),
    #    (p0, p1))

    #np.save('gm_256.npy', gm_extrapolated)
    #np.save('wm_256.npy', wm_extrapolated)
    #np.save('cortex_256.npy', cortex_extrapolated)

    return img_extrapolated, t1_extrapolated, T2short_ms_extrapolated, T2long_ms_extrapolated


def setup_blob_phantom(simulation_matrix_size: int):
    """simple central blob phantom to test normalization factor between nufft data and IFFT"""

    img_shape = 3 * (simulation_matrix_size, )

    x = np.linspace(-1, 1, simulation_matrix_size)
    X0, X1, X2 = np.meshgrid(x, x, x, indexing='ij')
    R = np.sqrt(X0**2 + X1**2 + X2**2)

    # create a central blob with sum() = 1
    img = np.zeros(img_shape)
    img[R < 0.25] = 1

    img = gaussian_filter(img, 2)

    # dummy proton T1 and Na T2star images
    t1 = img.copy()

    T2short = np.full(img_shape, 0.5 * np.finfo(np.float32).max)
    T2long = np.full(img_shape, 0.5 * np.finfo(np.float32).max)

    return img, t1, T2short, T2long


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

if __name__ == '__main__':

    #---------------------------------------------------------------------
    # input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--gradient_strength',
                        type=int,
                        default=16,
                        choices=[16, 24, 32, 48])
    parser.add_argument('--no_decay', action='store_true')
    parser.add_argument('--phantom',
                        choices=['brainweb', 'blob'],
                        default='brainweb')
    args = parser.parse_args()

    gradient_strength = args.gradient_strength
    no_decay = args.no_decay
    phantom = args.phantom

    simulation_matrix_size: int = 256
    field_of_view_cm: float = 22.
    phantom_data_path: Path = Path('/data/sodium_mr/brainweb54')

    if gradient_strength == 16:
        gradient_file: str = '/data/sodium_mr/tpi_gradients/n28p4dt10g16_23Na_v1'
    elif gradient_strength == 24:
        gradient_file: str = '/data/sodium_mr/tpi_gradients/n28p4dt10g24f23'
    elif gradient_strength == 32:
        gradient_file: str = '/data/sodium_mr/tpi_gradients/n28p4dt10g32f23'
    elif gradient_strength == 48:
        gradient_file: str = '/data/sodium_mr/tpi_gradients/n28p4dt10g48f23'
    else:
        raise ValueError

    # the number of time steps for the data simulation
    num_time_bins: int = 300

    # the two echo times in ms
    t_echo_1_ms: float = 0.5
    t_echo_2_ms: float = 5.

    gridded_data_matrix_size: int = 128

    if no_decay:
        decay_suffix = '_no_decay'
        T2long_ms_csf: float = 1e7
        T2long_ms_gm: float = 1e7
        T2long_ms_wm: float = 1e7
        T2short_ms_csf: float = 1e7
        T2short_ms_gm: float = 1e7
        T2short_ms_wm: float = 1e7
    else:
        decay_suffix = ''
        T2long_ms_csf: float = 50.
        T2long_ms_gm: float = 15.
        T2long_ms_wm: float = 18.
        T2short_ms_csf: float = 50.
        T2short_ms_gm: float = 8.
        T2short_ms_wm: float = 9.

    #---------------------------------------------------------------------
    #---------------------------------------------------------------------
    #---------------------------------------------------------------------

    # (1) setup the brainweb phantom with the given simulation matrix size
    if phantom == 'brainweb':
        na_image, t1_image, T2short_ms, T2long_ms = setup_brainweb_phantom(
            simulation_matrix_size,
            phantom_data_path,
            field_of_view_cm=field_of_view_cm,
            T2long_ms_csf=T2long_ms_csf,
            T2long_ms_gm=T2long_ms_gm,
            T2long_ms_wm=T2long_ms_wm,
            T2short_ms_csf=T2short_ms_csf,
            T2short_ms_gm=T2short_ms_gm,
            T2short_ms_wm=T2short_ms_wm)
    elif phantom == 'blob':
        na_image, t1_image, T2short_ms, T2long_ms = setup_blob_phantom(
            simulation_matrix_size)
    else:
        raise ValueError

    output_path = Path(
        '/data'
    ) / 'sodium_mr' / f'{phantom}_{Path(gradient_file).name}{decay_suffix}'
    output_path.mkdir(exist_ok=True)

    #---------------------------------------------------------------------
    #---------------------------------------------------------------------
    # (2) read the TPI kspace trajectory from a gradient file

    # calculate the max kvalue for a 64x64x64 image with a FOV of 220
    # the max. k value is equal to 1 / (2*pixelsize) = 1 / (2*FOV/matrix_size)
    kmax = 1 / (2 * field_of_view_cm / 64)

    # read the k-space trajectories from file
    # they have physical units 1/cm
    kx, ky, kz, header, n_readouts_per_cone = read_tpi_gradient_files(
        gradient_file)
    #show_tpi_readout(kx, ky, kz, header, n_readouts_per_cone)

    # the gradient files only contain a half sphere
    # we add the 2nd half where all gradients are reversed
    kx = np.vstack((kx, -kx))
    ky = np.vstack((ky, -ky))
    kz = np.vstack((kz, -kz))
    correct_tpi_sampling_density = True

    #----------------------------------------------------------------------------
    #-- calculate a NUFFT of a simple test image --------------------------------
    #----------------------------------------------------------------------------

    # split the array of all times points into a number of subsets
    time_bins_inds = np.array_split(np.arange(kx.shape[1]), num_time_bins)

    nuffts = []
    k0 = []
    k1 = []
    k2 = []

    nonuniform_data_long_echo_1 = []
    nonuniform_data_long_echo_2 = []

    nonuniform_data_short_echo_1 = []
    nonuniform_data_short_echo_2 = []

    nufft_device = pynufft.helper.device_list()[0]
    nufft_3d = pynufft.NUFFT(nufft_device)

    # loop over discretized time interval
    # calculate T2star decay and kspace points that are read out
    # in a given time bin
    for i, time_bin_inds in enumerate(time_bins_inds):
        kspace_sample_points = np.zeros((kx.shape[0] * time_bin_inds.size, 3),
                                        dtype=np.float32)
        kspace_sample_points[:, 0] = kx[:, time_bin_inds].ravel()
        kspace_sample_points[:, 1] = ky[:, time_bin_inds].ravel()
        kspace_sample_points[:, 2] = kz[:, time_bin_inds].ravel()

        # kspace points that we need later for the regridding
        k0.append(kspace_sample_points[:, 0].copy())
        k1.append(kspace_sample_points[:, 1].copy())
        k2.append(kspace_sample_points[:, 2].copy())

        # for the NUFFT the nominal kmax needs to be scale to pi
        # remember that the kmax calculated above is for a 64,64,64 grid
        # if we use a different matrix size, we have to adjust kmax
        kspace_sample_points *= (np.pi / (kmax * simulation_matrix_size / 64))

        print(f'setting up NUFFT {(i+1):03}/{num_time_bins:03}')
        nufft_3d.plan(kspace_sample_points, 3 * (simulation_matrix_size, ),
                      3 * (2 * simulation_matrix_size, ), 3 * (6, ))

        # setup the readout times in ms of the first and second echo
        # the gradient data is sampled in 10 micro second steps, so we have to
        # divide by 100 to get the time in ms
        t_readout_echo_1_ms = t_echo_1_ms + (time_bins_inds[i][0] / 100)
        t_readout_echo_2_ms = t_echo_2_ms + (time_bins_inds[i][0] / 100)

        # calculate the decayed images at the two echo times for the fast and slow decay
        decayed_image_long_echo_1 = na_image * np.exp(
            -t_readout_echo_1_ms / T2long_ms)
        decayed_image_long_echo_2 = na_image * np.exp(
            -t_readout_echo_2_ms / T2long_ms)

        decayed_image_short_echo_1 = na_image * np.exp(
            -t_readout_echo_1_ms / T2short_ms)
        decayed_image_short_echo_2 = na_image * np.exp(
            -t_readout_echo_2_ms / T2short_ms)

        # calculate the nuffts of the decayed images
        nonuniform_data_long_echo_1.append(
            nufft_3d.forward(decayed_image_long_echo_1))
        nonuniform_data_long_echo_2.append(
            nufft_3d.forward(decayed_image_long_echo_2))
        nonuniform_data_short_echo_1.append(
            nufft_3d.forward(decayed_image_short_echo_1))
        nonuniform_data_short_echo_2.append(
            nufft_3d.forward(decayed_image_short_echo_2))

    # convert the nufft data from list into 1D array
    nonuniform_data_long_echo_1 = np.concatenate(nonuniform_data_long_echo_1)
    nonuniform_data_long_echo_2 = np.concatenate(nonuniform_data_long_echo_2)
    nonuniform_data_short_echo_1 = np.concatenate(nonuniform_data_short_echo_1)
    nonuniform_data_short_echo_2 = np.concatenate(nonuniform_data_short_echo_2)

    k0 = np.concatenate(k0)
    k1 = np.concatenate(k1)
    k2 = np.concatenate(k2)

    #---------------------------------------------------------------
    # the k-value in 1/cm at which the trajectories start twisting
    # we need that for the smapling density correction in the regridding later
    kp: float = 0.4 * 18 / field_of_view_cm

    # save the images and the nufft data to a file since the generation takes time
    np.savez(output_path / 'simulated_nufft_data.npz',
             nonuniform_data_long_echo_1=nonuniform_data_long_echo_1,
             nonuniform_data_long_echo_2=nonuniform_data_long_echo_2,
             nonuniform_data_short_echo_1=nonuniform_data_short_echo_1,
             nonuniform_data_short_echo_2=nonuniform_data_short_echo_2,
             k0=k0,
             k1=k1,
             k2=k2,
             kp=kp,
             kmax=kmax,
             kx=kx,
             ky=ky,
             kz=kz,
             t_echo_1_ms=t_echo_1_ms,
             t_echo_2_ms=t_echo_2_ms,
             field_of_view_cm=field_of_view_cm,
             na_image=na_image,
             t1_image=t1_image,
             T2short_ms=T2short_ms,
             T2long_ms=T2long_ms)

    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------
    # regrid and IFFT echo 1 data as check

    cutoff: float = 1.
    window: npt.NDArray = np.ones(100)

    # allocated memory for output arrays
    sampling_weights: npt.NDArray = np.zeros(
        (gridded_data_matrix_size, gridded_data_matrix_size,
         gridded_data_matrix_size),
        dtype=complex)
    regridded_data_echo_1: npt.NDArray = np.zeros(
        (gridded_data_matrix_size, gridded_data_matrix_size,
         gridded_data_matrix_size),
        dtype=complex)
    regridded_data_echo_2: npt.NDArray = np.zeros(
        (gridded_data_matrix_size, gridded_data_matrix_size,
         gridded_data_matrix_size),
        dtype=complex)

    print('calculating weights')
    regrid_tpi_data(gridded_data_matrix_size,
                    1 / field_of_view_cm,
                    nonuniform_data_long_echo_1,
                    k0.size,
                    k0.ravel(),
                    k1.ravel(),
                    k2.ravel(),
                    kmax,
                    kp,
                    window,
                    cutoff,
                    sampling_weights,
                    correct_tpi_sampling_density=correct_tpi_sampling_density,
                    output_weights=True)

    print('regridding data')
    regrid_tpi_data(gridded_data_matrix_size,
                    1 / field_of_view_cm,
                    0.6 * nonuniform_data_short_echo_1 +
                    0.4 * nonuniform_data_long_echo_1,
                    k0.size,
                    k0.ravel(),
                    k1.ravel(),
                    k2.ravel(),
                    kmax,
                    kp,
                    window,
                    cutoff,
                    regridded_data_echo_1,
                    correct_tpi_sampling_density=correct_tpi_sampling_density,
                    output_weights=False)

    print('IFFT recon')
    # don't forget to fft shift the data since the regridding function puts the kspace
    # origin in the center of the array
    regridded_data_echo_1 = np.fft.fftshift(regridded_data_echo_1)

    # numpy's fft handles the phase factor of the DFT diffrently compared to pynufft
    # so we have to apply a phase factor to the regridded data
    # in 1D this phase factor is [1,-1,1,-1, ...]
    # in 3D it is the 3D checkerboard version of this
    # see here for details https://stackoverflow.com/questions/24077913/discretized-continuous-fourier-transform-with-numpy
    tmp_x = np.arange(gridded_data_matrix_size)
    TMP_X, TMP_Y, TMP_Z = np.meshgrid(tmp_x, tmp_x, tmp_x)

    phase_correction = ((-1)**TMP_X) * ((-1)**TMP_Y) * ((-1)**TMP_Z)
    regridded_data_echo_1_phase_corrected = phase_correction * regridded_data_echo_1

    # IFFT of the regridded data
    ifft_echo_1 = np.fft.ifftn(regridded_data_echo_1_phase_corrected,
                               norm='ortho')

    # the regridding in kspace uses trilinear interpolation (convolution with a triangle)
    # we the have to divide by the FT of a triangle (sinc^2)
    tmp_x = np.linspace(-0.5, 0.5, gridded_data_matrix_size)
    TMP_X, TMP_Y, TMP_Z = np.meshgrid(tmp_x, tmp_x, tmp_x)

    # corretion field is sinc(R)**2
    corr_field = np.sinc(np.sqrt(TMP_X**2 + TMP_Y**2 + TMP_Z**2))**2

    ifft_echo_1_corr = ifft_echo_1 / corr_field

    # interpolate magnitude images to simulation grid size (which can be different from gridded data size)
    a = zoom3d(np.abs(ifft_echo_1),
               simulation_matrix_size / gridded_data_matrix_size)
    b = zoom3d(np.abs(ifft_echo_1_corr),
               simulation_matrix_size / gridded_data_matrix_size)

    vi = pv.ThreeAxisViewer([na_image, a, b])

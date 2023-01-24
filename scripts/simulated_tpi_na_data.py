"""script to regrid 3D TPI (twisted projection) k-space data"""

import numpy as np
import math
import numpy.typing as npt
from numba import jit
import matplotlib.pyplot as plt


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

if __name__ == '__main__':

    matrix_size: int = 256
    field_of_view: float = 22.
    # calculate the max kvalue for a 64x64x64 image with a FOV of 220
    # the max. k value is equal to 1 / (2*pixelsize) = 1 / (2*FOV/matrix_size)
    kmax = 1 / (2 * field_of_view / 64)

    # read the k-space trajectories from file
    # they have physical units 1/cm
    kx, ky, kz, header, n_readouts_per_cone = read_tpi_gradient_files(
        '/data/tpi_gradients/n28p4dt10g16_23Na_v1')
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

    import pynufft

    tmp = np.linspace(-1, 1, matrix_size)
    TMP0, TMP1, TMP2 = np.meshgrid(tmp, tmp, tmp)
    R = np.sqrt(TMP0**2 + TMP1**2 + TMP2**2)
    img = (R <= 0.5).astype(np.float32)
    img[R <= 0.25] = 2.

    # split the array of all times points into a number of subsets
    num_time_bins = 200
    time_bins_inds = np.array_split(np.arange(kx.shape[1]), num_time_bins)

    nuffts = []
    k0 = []
    k1 = []
    k2 = []

    nonuniform_data = []
    T2star = np.full(img.shape, 40, dtype=np.float32)
    T2star[img == 2] = 8.

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
        kspace_sample_points *= (np.pi / (kmax * matrix_size / 64))

        print(f'setting up NUFFT {(i+1):03}/{num_time_bins:03}')
        nufft_3d.plan(kspace_sample_points, 3 * (matrix_size, ),
                      3 * (2 * matrix_size, ), 3 * (6, ))

        t_readout = 0.5 + (time_bins_inds[i][0] / 100)

        decayed_image = img * np.exp(-t_readout / T2star)
        nonuniform_data.append(nufft_3d.forward(decayed_image))

    nonuniform_data = np.concatenate(nonuniform_data)
    k0 = np.concatenate(k0)
    k1 = np.concatenate(k1)
    k2 = np.concatenate(k2)

    #---------------------------------------------------------------
    # the k-value in 1/cm at which the trajectories start twisting
    # we need that for the smapling density correction in the regridding later
    kp: float = 0.4 * 18 / field_of_view

    cutoff: float = 1.
    window: npt.NDArray = np.ones(100)

    # allocated memory for output arrays
    sampling_weights: npt.NDArray = np.zeros(
        (matrix_size, matrix_size, matrix_size), dtype=complex)
    regridded_data: npt.NDArray = np.zeros(
        (matrix_size, matrix_size, matrix_size), dtype=complex)

    print('calculating weights')
    regrid_tpi_data(matrix_size,
                    1 / field_of_view,
                    nonuniform_data,
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
    regrid_tpi_data(matrix_size,
                    1 / field_of_view,
                    nonuniform_data,
                    k0.size,
                    k0.ravel(),
                    k1.ravel(),
                    k2.ravel(),
                    kmax,
                    kp,
                    window,
                    cutoff,
                    regridded_data,
                    correct_tpi_sampling_density=correct_tpi_sampling_density,
                    output_weights=False)

    print('IFFT recon')
    # don't forget to fft shift the data since the regridding function puts the kspace
    # origin in the center of the array
    regridded_data = np.fft.fftshift(regridded_data)

    # IFFT of the regridded data
    ifft_recon = np.fft.fftshift(np.fft.ifftn(regridded_data))

    # the regridding in kspace uses trilinear interpolation (convolution with a triangle)
    # we the have to divide by the FT of a triangle (sinc^2)
    tmp_x = np.linspace(-0.5, 0.5, matrix_size)
    TMP_X, TMP_Y, TMP_Z = np.meshgrid(tmp_x, tmp_x, tmp_x)

    # corretion field is sinc(R)**2
    corr_field = np.sinc(np.sqrt(TMP_X**2 + TMP_Y**2 + TMP_Z**2))**2

    ifft_recon_corr = ifft_recon / corr_field

    import pymirc.viewer as pv
    vi = pv.ThreeAxisViewer([img, np.abs(ifft_recon), np.abs(ifft_recon_corr)])

import math
import numpy as np
import numpy.typing as npt
from numba import jit
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import nibabel as nib
import math
from typing import Union

from pymirc.image_operations import aff_transform, zoom3d


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


def tpi_sampling_density(kx: npt.NDArray, ky: npt.NDArray, kz: npt.NDArray,
                         kp: float):
    kabs = np.sqrt(kx**2 + ky**2 + kz**2)

    sampling_density = (kabs**2)
    sampling_density[kabs > kp] = kp**2

    return sampling_density


@jit(nopython=True)
def trilinear_kspace_interpolation(non_uniform_data, kx, ky, kz, matrix_size,
                                   delta_k, kmax, output):

    for i in range(kx.size):
        kabs = np.sqrt(kx[i]**2 + ky[i]**2 + kz[i]**2)
        if kabs <= kmax:

            kx_shifted = (kx[i] / delta_k) + 0.5 * (matrix_size)
            ky_shifted = (ky[i] / delta_k) + 0.5 * (matrix_size)
            kz_shifted = (kz[i] / delta_k) + 0.5 * (matrix_size)

            kx_shifted_low = math.floor(kx_shifted)
            ky_shifted_low = math.floor(ky_shifted)
            kz_shifted_low = math.floor(kz_shifted)

            kx_shifted_high = kx_shifted_low + 1
            ky_shifted_high = ky_shifted_low + 1
            kz_shifted_high = kz_shifted_low + 1

            dkx = float(kx_shifted - kx_shifted_low)
            dky = float(ky_shifted - ky_shifted_low)
            dkz = float(kz_shifted - kz_shifted_low)

            toAdd = non_uniform_data[i]

            if (kx_shifted_low >= 0) and (ky_shifted_low >=
                                          0) and (kz_shifted_low >= 0):

                output[kx_shifted_low, ky_shifted_low,
                       kz_shifted_low] += (1 - dkx) * (1 - dky) * (1 -
                                                                   dkz) * toAdd

                output[kx_shifted_high, ky_shifted_low,
                       kz_shifted_low] += (dkx) * (1 - dky) * (1 - dkz) * toAdd

                output[kx_shifted_low, ky_shifted_high,
                       kz_shifted_low] += (1 - dkx) * (dky) * (1 - dkz) * toAdd

                output[kx_shifted_high, ky_shifted_high,
                       kz_shifted_low] += (dkx) * (dky) * (1 - dkz) * toAdd

                output[kx_shifted_low, ky_shifted_low,
                       kz_shifted_high] += (1 - dkx) * (1 -
                                                        dky) * (dkz) * toAdd

                output[kx_shifted_high, ky_shifted_low,
                       kz_shifted_high] += (dkx) * (1 - dky) * (dkz) * toAdd

                output[kx_shifted_low, ky_shifted_high,
                       kz_shifted_high] += (1 - dkx) * (dky) * (dkz) * toAdd

                output[kx_shifted_high, ky_shifted_high,
                       kz_shifted_high] += (dkx) * (dky) * (dkz) * toAdd


class TriliniearKSpaceRegridder:

    def __init__(self,
                 matrix_size: int,
                 delta_k: float,
                 kx: npt.NDArray,
                 ky: npt.NDArray,
                 kz: npt.NDArray,
                 sampling_density: npt.NDArray,
                 kmax: Union[float, None] = None,
                 phase_correct: bool = True,
                 normalize_central_weight=True) -> None:
        self._matrix_size = matrix_size
        self._delta_k = delta_k
        self._phase_correct = phase_correct
        self._normalize_central_weight = normalize_central_weight

        self._kx = kx
        self._ky = ky
        self._kz = kz

        self._kabs = np.sqrt(kx**2 + ky**2 + kz**2)

        self._sampling_density = sampling_density.astype(np.float64)

        if kmax is None:
            self._kmax = np.max(self.kabs)
        else:
            self._kmax = kmax

        self._sampling_weights = np.zeros(
            (self.matrix_size, self.matrix_size, self.matrix_size),
            dtype=np.float64)

        # calculate the sampling weights which we need to correct for the
        # if we want to correct for the sampling density in the center
        trilinear_kspace_interpolation(self._sampling_density, self._kx,
                                       self._ky, self._kz, self._matrix_size,
                                       self._delta_k, self._kmax,
                                       self._sampling_weights)

        self._central_weight = self._sampling_weights[self._matrix_size // 2,
                                                      self._matrix_size // 2,
                                                      self._matrix_size // 2]

        if self._normalize_central_weight:
            self._sampling_weights /= self._central_weight

        # create the phase correction field containing a checkerboard pattern
        tmp_x = np.arange(self._matrix_size)
        TMP_X, TMP_Y, TMP_Z = np.meshgrid(tmp_x, tmp_x, tmp_x)

        self._phase_correction_field = ((-1)**TMP_X) * ((-1)**TMP_Y) * (
            (-1)**TMP_Z)

        # create the correction field due to interpolation (FT of interpolation kernel)
        tmp_x = np.linspace(-0.5, 0.5, self._matrix_size)
        TMP_X, TMP_Y, TMP_Z = np.meshgrid(tmp_x, tmp_x, tmp_x)
        self._interpolation_correction_field = np.sinc(
            np.sqrt(TMP_X**2 + TMP_Y**2 + TMP_Z**2))**2

    @property
    def kabs(self) -> npt.NDArray:
        return self._kabs

    @property
    def matrix_size(self) -> int:
        return self._matrix_size

    @property
    def delta_k(self) -> float:
        return self._delta_k

    @property
    def kx(self) -> npt.NDArray:
        return self._kx

    @property
    def ky(self) -> npt.NDArray:
        return self._ky

    @property
    def kz(self) -> npt.NDArray:
        return self._kz

    @property
    def sampling_density(self) -> npt.NDArray:
        return self._sampling_density

    @property
    def kmax(self) -> float:
        return self._kmax

    @property
    def sampling_weights(self) -> npt.NDArray:
        return self._sampling_weights

    @property
    def central_weight(self) -> float:
        return self._central_weight

    @property
    def phase_correct(self) -> bool:
        return self._phase_correct

    @property
    def phase_correction_field(self) -> npt.NDArray:
        return self._phase_correction_field

    @property
    def interpolation_correction_field(self) -> npt.NDArray:
        return self._interpolation_correction_field

    def regrid(self, non_uniform_data: npt.NDArray) -> npt.NDArray:
        output = np.zeros(
            (self.matrix_size, self.matrix_size, self.matrix_size),
            dtype=non_uniform_data.dtype)

        trilinear_kspace_interpolation(
            self._sampling_density * non_uniform_data, self._kx, self._ky,
            self._kz, self._matrix_size, self._delta_k, self._kmax, output)

        output = np.fft.fftshift(output)

        if self._phase_correct:
            output *= self._phase_correction_field

        if self._normalize_central_weight:
            output /= self._central_weight

        # correct for matrix size, important when we use ffts with norm = 'ortho'
        output /= np.sqrt((128 / self._matrix_size)**3)

        # correct for fall-off towards the edges due to interpolation
        ifft_output = np.fft.ifftn(
            output, norm='ortho') / self._interpolation_correction_field
        output = np.fft.fftn(ifft_output, norm='ortho')

        return output

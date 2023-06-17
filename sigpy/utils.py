import math
import numpy as np
import numpy.typing as npt
from numba import jit
from pathlib import Path
import cupy as cp

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import nibabel as nib
import math
from typing import Union, Sequence

from pymirc.image_operations import aff_transform, zoom3d
#from pynamr.utils import TPIReadOutTime, RadialKSpacePartitioner

import SimpleITK as sitk


def kb_rolloff(x: np.ndarray, beta: float):
    """roll-off due to Kaiser-Bessel window
       see Jackson et al IEEE TMI 1991 Selection of a Conv. Func for Fourier Inverse Regridding  Eq.15

       Parameters
       ----------

       x: np.ndarray
          normalized spatial coordinates (x*W in Jackson paper)
          for a window width of 2*delta_k and delta_k = 1/FOV, x should be in the range [-1, 1]
       beta: float
          beta parameter of Kaiser-Bessel window
    """
    y = (np.pi**2) * (x**2) - beta**2

    i0 = np.where(y > 0)
    i1 = np.where(y < 0)

    z0 = np.sqrt(y[i0])
    z1 = np.sqrt(-y[i1])

    res = np.ones_like(x)

    res[i0] = np.sin(z0) / z0
    res[i1] = np.sinh(z1) / z1

    return res


def hann(k: np.ndarray, k0: float):
    res = np.zeros_like(k)
    inds = np.where(k <= k0)
    res[inds] = 0.5 * (1 + np.cos(np.pi * k[inds] / k0))

    return res


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


def setup_brainweb_phantom(
    simulation_matrix_size: int,
    phantom_data_path: Path,
    field_of_view_cm: float = 22.,
    csf_na_concentration: float = 3.0,
    gm_na_concentration: float = 1.5,
    wm_na_concentration: float = 1.0,
    other_na_concentration: float = 0.75,
    T2long_ms_csf: float = 50.,
    T2long_ms_gm: float = 15.,
    T2long_ms_wm: float = 18.,
    T2long_ms_other: float = 15.,
    T2short_ms_csf: float = 50.,
    T2short_ms_gm: float = 8.,
    T2short_ms_wm: float = 9.,
    T2short_ms_other: float = 8.,
    add_anatomical_mismatch: bool = False,
    add_T2star_bias: bool = False,
    pathology:
    str = 'none',  # none or combinations of big-lesion, small-lesion, cos, gm, wm, gwm
    pathology_change_perc: float = 0.):

    simulation_voxel_size_mm: float = 10 * field_of_view_cm / simulation_matrix_size

    # setup the phantom on a high resolution grid (0.5^3mm) first
    label_nii = nib.load(phantom_data_path / 'subject54_crisp_v.nii')
    label_nii = nib.as_closest_canonical(label_nii)

    # pad to 220mm FOV
    lab_voxelsize = label_nii.header['pixdim'][1]
    lab = label_nii.get_fdata()
    pad_size_220 = ((220 - np.array(lab.shape) * lab_voxelsize) /
                    lab_voxelsize / 2).astype(int)
    pad_size_220 = ((pad_size_220[0], pad_size_220[0]),
                    (pad_size_220[1], pad_size_220[1]), (pad_size_220[2],
                                                         pad_size_220[2]))
    lab = np.pad(lab, pad_size_220, 'constant')

    # CSF = 1, GM = 2, WM = 3
    csf_inds = np.where(lab == 1)
    gm_inds = np.where(lab == 2)
    wm_inds = np.where(lab == 3)
    other_inds = np.where(lab >= 4)
    skull_inds = np.where(lab == 7)

    # calculate eye masks
    x = np.arange(lab.shape[0])
    X, Y, Z = np.meshgrid(x, x, x)
    R1 = np.sqrt((X - 368)**2 + (Y - 143)**2 + (Z - 97)**2)
    R2 = np.sqrt((X - 368)**2 + (Y - 291)**2 + (Z - 97)**2)
    eye1_inds = np.where((R1 < 25))
    eye2_inds = np.where((R2 < 25))

    # set up array for trans. magnetization
    img = np.zeros(lab.shape, dtype=np.float32)
    img[csf_inds] = csf_na_concentration
    img[gm_inds] = gm_na_concentration
    img[wm_inds] = wm_na_concentration
    img[other_inds] = other_na_concentration
    img[skull_inds] = 0.1
    img[eye1_inds] = csf_na_concentration
    img[eye2_inds] = csf_na_concentration

    # add pathology if required
    # only in the foreground, only for Na "concentration"
    if pathology != 'none' and pathology_change_perc > 0.:
        patho_mask = img > 0.
        patho_only = np.ones(img.shape, img.dtype)
        if 'gwm' in pathology:
            patho_mask *= (lab == 2) + (lab == 3)
        elif 'gm' in pathology:
            patho_mask *= (lab == 2)
        elif 'wm' in pathology:
            patho_mask *= (lab == 3)
        if 'cos' in pathology:
            cos_1d = cos_im_for_tpi(img.shape)
            patho_only = patho_only * np.broadcast_to(cos_1d, img.shape)
        if 'lesion' in pathology:
            if 'small' in pathology:
                radius = img.shape[0] // 20
                center = img.shape[0] // 2 + 2 * radius
            elif 'big' in pathology:
                radius = img.shape[0] // 10
                center = img.shape[0] // 2 + radius
            else:
                radius = img.shape[0] // 20
                center = img.shape[0] // 2 + radius
            # build indices for a sphere with given radius and center
            ind = np.arange(img.shape[0])
            k0, k1, k2 = np.meshgrid(ind, ind, ind)
            k_abs = np.sqrt((k0 - center)**2 + (k1 - center)**2 +
                            (k2 - center)**2)
            patho_mask *= k_abs < radius

        # the pathological sodium concentration is the maximum normal intensity in the same area
        # increased by the given percentage
        ref_intensity = np.max(img[patho_mask])
        patho_intensity = ref_intensity * pathology_change_perc / 100.

        # add the pathology only in the mask area
        img[patho_mask] += patho_only[patho_mask] * patho_intensity

    # set up array for Gamma (ratio between 2nd and 1st echo)
    T2short_ms = np.full(lab.shape,
                         0.5 * np.finfo(np.float32).max,
                         dtype=np.float32)
    T2short_ms[csf_inds] = T2short_ms_csf
    T2short_ms[gm_inds] = T2short_ms_gm
    T2short_ms[wm_inds] = T2short_ms_wm
    T2short_ms[other_inds] = T2short_ms_other
    T2short_ms[eye1_inds] = T2short_ms_csf
    T2short_ms[eye2_inds] = T2short_ms_csf

    T2long_ms = np.full(lab.shape,
                        0.5 * np.finfo(np.float32).max,
                        dtype=np.float32)
    T2long_ms[csf_inds] = T2long_ms_csf
    T2long_ms[gm_inds] = T2long_ms_gm
    T2long_ms[wm_inds] = T2long_ms_wm
    T2long_ms[other_inds] = T2long_ms_other
    T2long_ms[eye1_inds] = T2long_ms_csf
    T2long_ms[eye2_inds] = T2long_ms_csf

    # read the T1
    t1_nii = nib.load(phantom_data_path / 'subject54_t1w_p4_resampled.nii')
    t1_nii = nib.as_closest_canonical(t1_nii)
    t1 = np.pad(t1_nii.get_fdata(), pad_size_220, 'constant')

    # add eye contrast
    t1[eye1_inds] *= 0.5
    t1[eye2_inds] *= 0.5

    # add mismatches
    if add_anatomical_mismatch:
        R1 = np.sqrt((X - 329)**2 + (Y - 165)**2 + (Z - 200)**2)
        inds1 = np.where((R1 < 10))
        img[inds1] = gm_na_concentration
        #R2 = np.sqrt((X - 327)**2 + (Y - 262)**2 + (Z - 200)**2)
        #inds2 = np.where((R2 < 10))
        #t1[inds2] = 0

    # add bias field on T2* times
    if add_T2star_bias:
        T2starbias = np.arctan((Z - 155) / 10) / (2 * np.pi) + 0.75
        T2short_ms *= T2starbias
        T2long_ms *= T2starbias

    # extrapolate the all images to the voxel size we need for the data simulation
    img_extrapolated = zoom3d(img, lab_voxelsize / simulation_voxel_size_mm)
    T2short_ms_extrapolated = zoom3d(T2short_ms,
                                     lab_voxelsize / simulation_voxel_size_mm)
    T2long_ms_extrapolated = zoom3d(T2long_ms,
                                    lab_voxelsize / simulation_voxel_size_mm)
    t1_extrapolated = zoom3d(t1, lab_voxelsize / simulation_voxel_size_mm)

    return img_extrapolated, t1_extrapolated, T2short_ms_extrapolated, T2long_ms_extrapolated


def setup_blob_phantom(
    simulation_matrix_size: int,
    radius: float = 0.25,
    T2short: float = 0.5 * np.finfo(np.float32).max,
    T2long: float = 0.5 * np.finfo(np.float32).max,
    longerT2ring: bool = False,
    pathology:
    str = 'none',  # none, cos, big-lesion, small-lesion, or combinations thereof
    pathology_change_perc: float = 0.):
    """simple central blob phantom to test normalization factor between nufft data and IFFT
        and study other effects (ideal observer analysis, influence of T2 nonuniformity, etc.)"""

    img_shape = 3 * (simulation_matrix_size, )

    x = np.linspace(-1, 1, simulation_matrix_size)
    X0, X1, X2 = np.meshgrid(x, x, x, indexing='ij')
    R = np.sqrt(X0**2 + X1**2 + X2**2)

    # create a central blob with sum() = 1
    img = np.zeros(img_shape)
    img[R < radius] = 1

    img = gaussian_filter(img, 2)

    # dummy proton T1 and Na T2star images
    t1 = img.copy()

    T2short_im = np.full(img_shape, T2short)
    T2long_im = np.full(img_shape, T2long)

    if longerT2ring:
        # add a ring of CSF like T2* values inside the cylinder
        inner = radius * 0.8
        T2short_im[np.logical_and(R < radius, R
                                  > inner)] = 50  #0.5*np.finfo(np.float32).max
        T2long_im[np.logical_and(R < radius, R
                                 > inner)] = 50  #0.5*np.finfo(np.float32).max

    # add pathology if required
    # only in the foreground, only for Na "concentration"
    if pathology != 'none' and pathology_change_perc > 0.:
        patho_mask = img > 0.
        patho_only = np.ones(img.shape, img.dtype)
        if 'cos' in pathology:
            cos_1d = cos_im_for_tpi(img.shape)
            patho_only = patho_only * np.broadcast_to(cos_1d, img.shape)
        if 'lesion' in pathology:
            if 'small' in pathology:
                radius = img.shape[0] // 20
                center = img.shape[0] // 2 + 2 * radius
            elif 'big' in pathology:
                radius = img.shape[0] // 10
                center = img.shape[0] // 2 + radius
            else:
                radius = img.shape[0] // 20
                center = img.shape[0] // 2 + radius

            # build indices for a sphere with given radius and center
            ind = np.arange(img.shape[0])
            k0, k1, k2 = np.meshgrid(ind, ind, ind)
            k_abs = np.sqrt((k0 - center)**2 + (k1 - center)**2 +
                            (k2 - center)**2)
            patho_mask *= k_abs < radius

        # the pathological sodium concentration is the maximum normal intensity in the same area
        # increased by the given percentage
        ref_intensity = np.max(img[patho_mask])
        patho_intensity = ref_intensity * pathology_change_perc / 100.

        # add the pathology only in the mask area
        img[patho_mask] += patho_only[patho_mask] * patho_intensity

    return img, t1, T2short_im, T2long_im


def cos_im_for_tpi(im_shape: tuple) -> np.ndarray:
    """ Build 1D cosine based on hard coded maximum TPI frequency, field of view,
        and given input image cubic matrix size
    """

    kmax_1_cm = 1.45
    field_of_view_cm = 22.
    n = im_shape[-1]

    # spatial frequency of the cosine: N - 1 (one before maximum discretized frequency)
    real_freq = kmax_1_cm - 1. / field_of_view_cm  # 1/cm
    vox_size = field_of_view_cm / n
    period = 1. / real_freq
    period_nb_vox = period / vox_size
    # cosine frequency
    freq = 1. / period_nb_vox

    # 1d cosine in space
    cos_1d = np.cos(2 * np.pi * freq * np.linspace(0, n, n, endpoint=False))
    # cosine image
    cos_im = np.broadcast_to(cos_1d, im_shape)

    return cos_im


def crop_kspace_data(data: np.ndarray,
                     in_shape: tuple,
                     out_shape: tuple,
                     center: bool = False) -> np.ndarray:
    """ Crop k-space data to a smaller grid
        Option for providing centered or fftshifted k-space data
        Assuming fft with ortho norm is used """

    if center:
        cropped = data[in_shape[0] // 2 - out_shape[0] // 2:in_shape[0] // 2 +
                       out_shape[0] // 2, in_shape[1] // 2 -
                       out_shape[1] // 2:in_shape[1] // 2 + out_shape[1] // 2,
                       in_shape[2] // 2 - out_shape[2] // 2:in_shape[2] // 2 +
                       out_shape[2] // 2]
    else:
        temp = np.fft.fftshift(data)
        temp = temp[in_shape[0] // 2 - out_shape[0] // 2:in_shape[0] // 2 +
                    out_shape[0] // 2, in_shape[1] // 2 -
                    out_shape[1] // 2:in_shape[1] // 2 + out_shape[1] // 2,
                    in_shape[2] // 2 - out_shape[2] // 2:in_shape[2] // 2 +
                    out_shape[2] // 2]
        cropped = np.fft.fftshift(temp)

    # normalization factor for preserving the same image scale
    # assuming ortho norm was used for fft
    norm_factor = 1
    for d in range(len(in_shape)):
        norm_factor *= np.sqrt(out_shape[d] / in_shape[d])

    cropped *= norm_factor
    return cropped


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

            if (kx_shifted_low >= 0) and (ky_shifted_low
                                          >= 0) and (kz_shifted_low >= 0):

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
                 final_transform: Union[sitk.Transform, None] = None,
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


def read_GE_ak_wav(fname: str):
    """
    read GE waveforms stored to external file using Stanford format

    Parameters
    ----------
    fname : str

    Returns
    -------
    grad: np.ndarray
        Gradient waveforms in (T/m)
        shape (#pts/interleave,#interleaves,#groups)
        with: #groups = 2 for 2d-imaging, =3 for 3d-imaging
    bw: float
        full bandwidth (opuser0 = 2d3*oprbw)           [Hz]
    fov: float
        field-of-view (old=nucleus; new=1H equivalent) [m]
    desc: np.ndarray
        description string (256 chars)
    N: dict
         N.gpts   # input gradient pts/interleave
         N.groups # groups
         N.intl   # interleaves
         N.params # parameters
    params : np.ndarray
        header file parameters (scanner units)
        grad_type fov N.intl gmax N.gpts gdt N.kpts kdt 0 0 0
    """
    N = {}
    offset = 0

    desc = np.fromfile(fname, dtype=np.int8, offset=offset, count=256)
    offset += desc.size * desc.itemsize

    N["gpts"] = np.fromfile(fname,
                            dtype=np.dtype('>u2'),
                            offset=offset,
                            count=1)[0]
    offset += N["gpts"].size * N["gpts"].itemsize

    N["groups"] = np.fromfile(fname,
                              dtype=np.dtype('>u2'),
                              offset=offset,
                              count=1)[0]
    offset += N["groups"].size * N["groups"].itemsize

    N["intl"] = np.fromfile(fname,
                            dtype=np.dtype('>u2'),
                            offset=offset,
                            count=N["groups"])
    offset += N["intl"].size * N["intl"].itemsize

    N["params"] = np.fromfile(fname,
                              dtype=np.dtype('>u2'),
                              offset=256 + 4 + N["groups"] * 2,
                              count=1)[0]
    offset += N["params"].size * N["params"].itemsize

    params = np.fromfile(fname,
                         dtype=np.dtype('>f8'),
                         offset=offset,
                         count=N["params"])
    offset += params.size * params.itemsize

    wave = np.fromfile(fname, dtype=np.dtype('>i2'), offset=offset)
    offset += wave.size * wave.itemsize

    grad = np.swapaxes(wave.reshape((N["groups"], N["intl"][0], N["gpts"])), 0,
                       2)

    # set stop bit to 0
    grad[-1, ...] = 0

    # scale gradients to SI units (T/m)
    grad = (grad / 32767) * (params[3] / 100)

    # bandwidth in (Hz)
    bw = 1e6 / params[7]

    # (proton) field of view in (m)
    fov = params[1] / 100

    return grad, bw, fov, desc, N, params


def simpleForward_TPI_FFT(image: cp.ndarray,
                          T2map_short: cp.ndarray,
                          T2map_long: cp.ndarray,
                          acquired_grid_n: int,
                          echo_time_ms: float,
                          num_time_bins: int,
                          acc: float = 1.) -> [cp.ndarray, cp.ndarray]:
    """Simple forward TPI operator using FFT approximation, assuming bi exp decay

        Parameters
        ----------
        image : input image
        T2map_short: image of short T2* component
        T2map_long: image of long T2* component
        acquired_grid_n: matrix size, assuming cubic matrix
        echo_time_ms : float
        num_time_bins: int, number of time bins
        acc : time acceleration of the hard coded TPI with max grad=0.16G/cm

        Returns
        ----------
        data : k-space data as would be approximately acquired with TPI
        k_mask : spherical mask for actually acquired k-space samples

        """

    ishape = image.shape
    # hard coded equation for TPI readout time
    # as a function of the distance from k-space center
    t = TPIReadOutTime()
    pad_factor = ishape[0] // acquired_grid_n
    k = RadialKSpacePartitioner(ishape, num_time_bins, pad_factor)

    time_bins = t(k.k) / acc
    k_inds = k.k_inds
    k_mask = cp.asarray(k.kmask)

    data = cp.zeros(ishape, dtype=cp.complex128)
    for i, time_bin in enumerate(time_bins):
        #setup the decay image
        readout_time_ms = echo_time_ms + time_bin
        decay = 0.6 * cp.exp(-readout_time_ms / T2map_short) + 0.4 * cp.exp(
            -readout_time_ms / T2map_long)

        # fft and sum
        data[k_inds[i]] += cp.fft.fftn(image * decay, norm='ortho')[k_inds[i]]

    return data, k_mask

import math
import numpy as np
import numpy.typing as npt
from numba import jit
from pathlib import Path
import cupy as cp

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

import nibabel as nib
import math
from typing import Union, Sequence, Tuple

from pymirc.image_operations import aff_transform, zoom3d

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

    header = np.fromfile(
        gradient_file, dtype=np.int16, offset=0, count=num_header_elements)

    # number of cones
    num_cones = int(header[0])
    # number of points in a single readout
    num_points = int(header[1])

    # time sampling step in seconds
    dt = float(header[2]) * (1e-6)

    # maximum gradient strength in G/cm corresponds to max short value (2**15 - 1 = 32767
    max_gradient = float(header[3]) / 100

    # number of readouts per cone
    num_readouts_per_cone = np.fromfile(
        gradient_file,
        dtype=np.int16,
        offset=num_header_elements * 2,
        count=num_cones)

    gradient_array = np.fromfile(
        gradient_file,
        dtype=np.int16,
        offset=(num_header_elements + num_cones) * 2,
        count=num_cones * num_points).reshape(num_cones, num_points)

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

        phis = np.linspace(
            phi0s[i_cone],
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


def radial_goldenmean_kspace_coords_1_cm(
        gradient_strength: int = 16,
        k_max_1_cm: float = 1.45,  #1/cm
        dt: float = 1e-5,  #s
        nb_spokes: int = 10000,
        fill_to_grad08_readout_time: bool = False):
    """Build an array of k-space coordinates [1/cm] for a 3D golden-means radial pulse sequence

        Parameters
        ----------

        gradient_strength: int 
            TPI maximum gradient in e-2 G/cm
        k_max_1_cm: float
            maximum frequency
        dt: float
            ADC temporal sampling period
        nb_spokes: int
            number of spokes from the center to the edge of k-space

        Returns
        ----------

        k_1_cm: np.ndarray of shape (num_time_samples, num_readouts, num_spatial_dims)
            k-space coordinates with unit 1/cm
    """

    # 2D golden angles
    phi_1 = 0.4656
    phi_2 = 0.6823
    gamma_by_2pi = 1126.2
    grad_g_cm = gradient_strength / 100.

    dk = gamma_by_2pi * dt * grad_g_cm  #1/cm
    nb_samples_per_spoke = int(np.ceil(k_max_1_cm / dk))
    k_1_cm = np.zeros((nb_samples_per_spoke, nb_spokes // 2, 3), np.float64)

    r = np.linspace(dk, k_max_1_cm, num=nb_samples_per_spoke)
    # half sphere
    for s in range(nb_spokes // 2):
        azimuth_angle = 2 * np.pi * ((s * phi_2) % 1)
        polar_angle = np.arccos((s * phi_1) % 1)
        k_1_cm[:, s, :] = np.array([
            r * np.sin(polar_angle) * np.cos(azimuth_angle),
            r * np.sin(polar_angle) * np.sin(azimuth_angle),
            r * np.cos(polar_angle)
        ]).T

    # readout time kept the same as for grad 0.08G/cm (32ms)
    # filled by going back and forth along traj
    if fill_to_grad08_readout_time:
        if gradient_strength == 16:
            k_1_cm = np.concatenate([k_1_cm, k_1_cm[::-1, :, :]], axis=0)
        elif gradient_strength == 32:
            k_1_cm = np.concatenate([k_1_cm, k_1_cm[::-1, :, :], k_1_cm, k_1_cm[::-1, :, :]],
                                    axis=0)

    # add the other half of the sphere
    k_1_cm = np.concatenate([k_1_cm, -k_1_cm], axis=1)
    return k_1_cm


def radial_random_kspace_coords_1_cm(
        gradient_strength: int = 16,  #10-2 G/cm
        k_max_1_cm: float = 1.45,  #1/cm
        dt: float = 1e-5,  #s
        nb_spokes: int = 10000):
    """Build an array of k-space coordinates [1/cm] for a radial pulse sequence
        by sampling randomly from uniform distributions for the cosine of the polar angle and for the azimuthal angle

        Parameters
        ----------

        gradient_strength: int
            TPI maximum gradient in e-2 G/cm
        k_max_1_cm: float
            maximum frequency
        dt: float
            ADC temporal sampling period
        nb_spokes: int
            number of spokes from the center to the edge of k-space

        Returns
        ----------

        k_1_cm: np.ndarray of shape (num_time_samples, num_readouts, num_spatial_dims)
            k-space coordinates with unit 1/cm
    """

    gamma_by_2pi = 1126.2
    grad_g_cm = gradient_strength / 100.

    dk = gamma_by_2pi * dt * grad_g_cm  #1/cm
    nb_samples_per_spoke = int(np.ceil(k_max_1_cm / dk))
    k_1_cm = np.zeros((nb_samples_per_spoke, nb_spokes, 3), np.float64)

    r = np.linspace(dk, k_max_1_cm, num=nb_samples_per_spoke)
    # half sphere
    for s in range(nb_spokes):
        azimuth_angle = np.random.uniform(0, 2 * np.pi)
        polar_angle = np.arccos(np.random.uniform(-1, 1))
        k_1_cm[:, s, :] = np.array([
            r * np.sin(polar_angle) * np.cos(azimuth_angle),
            r * np.sin(polar_angle) * np.sin(azimuth_angle),
            r * np.cos(polar_angle)
        ]).T

    return k_1_cm


def radial_density_adapted_kspace_coords_1_cm(
        gradient_strength: int = 16,
        k_max_1_cm: float = 1.45,  #1/cm
        dt: float = 1e-5,  #s
        nb_spokes: int = 10000,
        p: float = 0.4):
    """Build an array of k-space coordinates [1/cm] for a 3D golden-means radial pulse sequence
       and density adapted sampling

        Parameters
        ----------

        gradient_strength: int
            TPI maximum gradient in e-2 G/cm, more convenient for a category
        k_max_1_cm: float
            maximum frequency
        dt: float
            ADC temporal sampling period
        nb_spokes: int
            number of spokes from the center to the edge of k-space
        p: float
            fraction of k_max sampled radially

        Returns
        ----------

        k_1_cm: np.ndarray of shape (num_time_samples, num_readouts, num_spatial_dims)
            k-space coordinates with unit 1/cm
    """

    # golden angles
    phi_1 = 0.4656
    phi_2 = 0.6823
    gamma_by_2pi = 1126.2
    grad_g_cm = gradient_strength / 100.

    dk = gamma_by_2pi * dt * grad_g_cm  # G/cm

    k_0 = p * k_max_1_cm
    t_0 = k_0 / (gamma_by_2pi * grad_g_cm)

    t_max = t_0 + (k_max_1_cm**3 - k_0**3) / (
        3 * gamma_by_2pi * k_0**2 * grad_g_cm)
    nb_samples_per_spoke = int(np.ceil(t_max / dt))
    t = np.linspace(dt, t_max, num=nb_samples_per_spoke)

    r_lin = np.linspace(dk, k_0, num=int(np.ceil(k_0 / dk)))
    r_da = np.power(
        3 * gamma_by_2pi * k_0**2 * grad_g_cm *
        (t[int(np.ceil(t_0 / dt)):] - t_0) + k_0**3, 1 / 3)
    r = np.concatenate([r_lin, r_da])
    k_1_cm = np.zeros((nb_samples_per_spoke, nb_spokes // 2, 3), np.float64)
    # half sphere
    for s in range(nb_spokes // 2):
        azimuth_angle = 2 * np.pi * ((s * phi_2) % 1)
        polar_angle = np.arccos((s * phi_1) % 1)
        k_1_cm[:, s, :] = np.array([
            r * np.sin(polar_angle) * np.cos(azimuth_angle),
            r * np.sin(polar_angle) * np.sin(azimuth_angle),
            r * np.cos(polar_angle)
        ]).T

    # add the other half of the sphere
    k_1_cm = np.concatenate([k_1_cm, -k_1_cm], axis=1)
    return k_1_cm


def tpi_kspace_coords_1_cm_scanner(
        gradient_strength: int = 16,
        data_root_dir: str = None,
        fill_to_grad16_readout_time: bool = False) -> np.ndarray:
    """Build an array of k-space coordinates [1/cm] for the spectrally weigthed TPI pulse sequence

        Parameters
        ----------

        gradient_strength: int
            TPI maximum gradient in 10^{-2} G/cm
        data_root_dir: str
            folder containing gradient trace files
        fill_to_grad16_readout_time : bool
            extend the trajectory (continue sampling) until reaching the same readout time as for grad 16

        Returns
        ----------

        k_1_cm: np.ndarray of shape (num_time_samples, num_readouts, num_spatial_dims)
            k-space coordinates with unit 1/cm
    """
    # find trajectory gradient file
    if gradient_strength in [16, 8, 4, 2]:
        gradient_file: str = str(
            Path(data_root_dir) / 'tpi_gradients/n28p4dt10g16_23Na_v1')
    elif gradient_strength == 24:
        gradient_file: str = str(
            Path(data_root_dir) / 'tpi_gradients/n28p4dt10g24f23')
    elif gradient_strength == 32 or gradient_strength == 64:
        gradient_file: str = str(
            Path(data_root_dir) / 'tpi_gradients/n28p4dt10g32f23')
    elif gradient_strength == 48:
        gradient_file: str = str(
            Path(data_root_dir) / 'tpi_gradients/n28p4dt10g48f23')
    else:
        raise ValueError

    # read the k-space trajectories from file
    # they have physical units 1/cm
    # kx.shape = (num_readouts, num_time_samples)
    kx, ky, kz, header, n_readouts_per_cone = read_tpi_gradient_files(
        gradient_file)
    #show_tpi_readout(kx, ky, kz, header, n_readouts_per_cone)

    # artificial gradient strength 8 based on 2x oversampling in time of the gradient 16 trace
    # because no gradient 8 trace available yet
    if gradient_strength in [16, 8, 4, 2]:
        num_time_pts = kx.shape[1]
        interp_time = interp1d(np.arange(num_time_pts), kx, axis=1)
        oversample = np.arange(0, num_time_pts - 0.9, gradient_strength / 16)
        kx = interp_time(oversample)
        interp_time = interp1d(np.arange(num_time_pts), ky, axis=1)
        ky = interp_time(oversample)
        interp_time = interp1d(np.arange(num_time_pts), kz, axis=1)
        kz = interp_time(oversample)
    # artificial gradient strength 64 based on 2x undersampling in time of the gradient 32 trace
    # because no gradient 64 trace available yet
    elif gradient_strength == 64:
        kx = kx[:, ::2]
        ky = ky[:, ::2]
        kz = kz[:, ::2]

    # group k-space coordinates
    k_1_cm = np.stack([kx, ky, kz], axis=-1)
    # reshape as (num_time_samples, num_readouts, space_dim)
    k_1_cm = np.transpose(k_1_cm, (1, 0, 2))

    # readout time kept the same as for grad 16
    # filled by going back and forth along traj
    if fill_to_grad16_readout_time:
        if gradient_strength == 32:
            k_1_cm = np.concatenate([k_1_cm, k_1_cm[::-1, :, :]], axis=0)
        elif gradient_strength == 48:
            k_1_cm = np.concatenate([k_1_cm, k_1_cm[::-1, :, :], k_1_cm],
                                    axis=0)

#            chunk = 400
#            repeats = k_1_cm.shape[0] * 2 // chunk
#            k_1_cm = np.concatenate([k_1_cm, np.repeat(k_1_cm[-chunk:], repeats, axis=0)],
#                                    axis=0)

#            end_points = k_1_cm[-1,:,:]
#            new_end_points = end_points * 2.
#            addition = np.linspace(end_points, new_end_points, 2 * k_1_cm.shape[0])
#            k_1_cm = np.concatenate([k_1_cm, addition], axis=0)

        elif gradient_strength == 64:
            k_1_cm = np.concatenate(
                [k_1_cm, k_1_cm[::-1, :, :], k_1_cm, k_1_cm[::-1, :, :]],
                axis=0)

    # the gradient files only contain a half sphere
    k_1_cm = np.concatenate([k_1_cm, -k_1_cm], axis=1)

    k_1_cm_abs = np.linalg.norm(k_1_cm, axis=2)
    #print(f'readout kmax .: {k_1_cm_abs.max():.2f} 1/cm')

    return k_1_cm


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
        str = 'none',  # none or relevant combinations of [lesion, cos, gm, wm, gwm, multi] or glioma
        pathology_size_perc:
        float = 0.,  # size in percentage of image size (first dimension), not relevant for glioma
        pathology_change_perc:
        float = 0.,  # change in percentage with respect to underlying normal tissue, not relevant for cosine part
        pathology_center_perc: np.ndarray = np.array(
            [0., 0., 0.])  # not relevant for glioma
):

    simulation_voxel_size_mm: float = 10 * field_of_view_cm / simulation_matrix_size

    # setup the phantom on a high resolution grid (0.5^3mm) first
    label_nii = nib.load(phantom_data_path / 'subject54_crisp_v.nii')
    label_nii = nib.as_closest_canonical(label_nii)

    # pad to 220mm FOV
    lab_voxelsize = label_nii.header['pixdim'][1]
    lab = label_nii.get_fdata()
    pad_size_220 = ((220 - np.array(lab.shape) * lab_voxelsize) / lab_voxelsize
                    / 2).astype(int)
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

    # set up array for Gamma (ratio between 2nd and 1st echo)
    T2short_ms = np.full(
        lab.shape, 0.5 * np.finfo(np.float32).max, dtype=np.float32)
    T2short_ms[csf_inds] = T2short_ms_csf
    T2short_ms[gm_inds] = T2short_ms_gm
    T2short_ms[wm_inds] = T2short_ms_wm
    T2short_ms[other_inds] = T2short_ms_other
    T2short_ms[eye1_inds] = T2short_ms_csf
    T2short_ms[eye2_inds] = T2short_ms_csf

    T2long_ms = np.full(
        lab.shape, 0.5 * np.finfo(np.float32).max, dtype=np.float32)
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

    # add pathology if required
    if pathology == 'glioma' and pathology_change_perc > 0.:
        glioma_mask = np.fromfile(phantom_data_path / 'glioma_labshape.raw',
                                  int).reshape(img.shape)
        img[glioma_mask == 1] = 0.9 * csf_na_concentration
        glioma_ring = pathology_change_perc * wm_na_concentration / 100
        img[glioma_mask == 2] = glioma_ring
        # modify T2 values, same for before and after treatment
        T2short_ms[glioma_mask == 1] = T2short_ms_csf
        T2long_ms[glioma_mask == 1] = T2long_ms_csf
        T2short_ms[glioma_mask == 2] = T2short_ms_wm
        T2long_ms[glioma_mask == 2] = T2long_ms_wm
    elif pathology != 'none':
        # pathology mask init
        patho_mask = img > 0.
        # pathology intensity init
        patho_only = np.ones(img.shape, img.dtype)
        if 'gwm' in pathology:
            patho_mask *= (lab == 2) + (lab == 3)
        elif 'gm' in pathology:
            patho_mask *= (lab == 2)
        elif 'wm' in pathology:
            patho_mask *= (lab == 3)
        if 'cos' in pathology:
            cos_im = cos_im_for_tpi(img.shape)
            patho_only = patho_only * cos_im
        if 'lesion' in pathology:
            if 'multi' in pathology:
                # many random spherical lesions
                nb_lesions = 20
                rng = np.random.default_rng(42)
                radius_init = 0.5 * pathology_size_perc * img.shape[0] / 100.
                radii = rng.random(nb_lesions) * 2. * radius_init
                # centers
                centers = rng.random(nb_lesions) * img.shape[0] // 3
                # mask containing all the lesions
                temp_mask = np.zeros_like(patho_mask)
                for l in range(nb_lesions):
                    k0, k1, k2 = np.meshgrid(
                        np.arange(img.shape[0]),
                        np.arange(img.shape[1]),
                        np.arange(img.shape[2]),
                        indexing='ij')
                    k_abs = np.sqrt((k0 - img.shape[0] // 2 - centers[l])**2 +
                                    (k1 - img.shape[1] // 2 - centers[l])**2 +
                                    (k2 - img.shape[2] // 2 - centers[l])**2)
                    temp_mask += k_abs < radii[l]
                patho_mask *= temp_mask
            else:
                center = tuple(
                    np.multiply(pathology_center_perc / 100.,
                                np.array(img.shape)))

                # spherical lesion
                radius = 0.5 * pathology_size_perc * img.shape[0] / 100.
                # mask for a sphere with given radius and center
                k0, k1, k2 = np.meshgrid(
                    np.arange(img.shape[0]),
                    np.arange(img.shape[1]),
                    np.arange(img.shape[2]),
                    indexing='ij')
                k_abs = np.sqrt((k0 - center[0])**2 + (k1 - center[1])**2 +
                                (k2 - center[2])**2)
                patho_mask *= k_abs < radius

        if 'cos' in pathology:
            # the amplitude of the cosine is chosen as the maximum amplitude
            # that does not make the image intensity become negative
            patho_intensity = np.min(img[patho_mask])
        else:
            # the pathological sodium concentration is the maximum normal concentration in the same area
            # increased by the given percentage
            ref_intensity = np.max(img[patho_mask])
            patho_intensity = ref_intensity * pathology_change_perc / 100.

        # add the pathology only in the mask area
        img[patho_mask] += patho_only[patho_mask] * patho_intensity

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


def setup_blob_phantom(simulation_matrix_size: int,
                       radius: float = 0.25,
                       T2short: float = 0.5 * np.finfo(np.float32).max,
                       T2long: float = 0.5 * np.finfo(np.float32).max):
    """simple central blob phantom to test normalization factor between nufft data and IFFT
        and study other effects ( influence of T2 nonuniformity, etc.)"""

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

    return img, t1, T2short_im, T2long_im


def cos_im_for_tpi(im_shape: tuple,
                   kmax_1_cm: float = 1.45,
                   field_of_view_cm: float = 22.,
                   nb_deltak_from_max: float = 6.) -> np.ndarray:
    """ Build cosine image based on a high frequency acquired with TPI

    Returns: single frequency cosine image with shape identical to input image shape
    """

    n = im_shape[-1]

    # spatial frequency of the cosine: N - nb_deltak_from_max
    real_freq = kmax_1_cm - nb_deltak_from_max / field_of_view_cm  # 1/cm
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
                       out_shape[1] // 2:in_shape[1] // 2 +
                       out_shape[1] // 2, in_shape[2] // 2 -
                       out_shape[2] // 2:in_shape[2] // 2 + out_shape[2] // 2]
    else:
        temp = np.fft.fftshift(data)
        temp = temp[in_shape[0] // 2 - out_shape[0] // 2:in_shape[0] // 2 +
                    out_shape[0] // 2, in_shape[1] // 2 - out_shape[1] //
                    2:in_shape[1] // 2 + out_shape[1] // 2, in_shape[2] // 2 -
                    out_shape[2] // 2:in_shape[2] // 2 + out_shape[2] // 2]
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

            if (kx_shifted_low >= 0) and (ky_shifted_low >=
                                          0) and (kz_shifted_low >= 0):

                output[kx_shifted_low, ky_shifted_low, kz_shifted_low] += (
                    1 - dkx) * (1 - dky) * (1 - dkz) * toAdd

                output[kx_shifted_high, ky_shifted_low, kz_shifted_low] += (
                    dkx) * (1 - dky) * (1 - dkz) * toAdd

                output[kx_shifted_low, ky_shifted_high, kz_shifted_low] += (
                    1 - dkx) * (dky) * (1 - dkz) * toAdd

                output[kx_shifted_high, ky_shifted_high, kz_shifted_low] += (
                    dkx) * (dky) * (1 - dkz) * toAdd

                output[kx_shifted_low, ky_shifted_low, kz_shifted_high] += (
                    1 - dkx) * (1 - dky) * (dkz) * toAdd

                output[kx_shifted_high, ky_shifted_low, kz_shifted_high] += (
                    dkx) * (1 - dky) * (dkz) * toAdd

                output[kx_shifted_low, ky_shifted_high, kz_shifted_high] += (
                    1 - dkx) * (dky) * (dkz) * toAdd

                output[kx_shifted_high, ky_shifted_high, kz_shifted_high] += (
                    dkx) * (dky) * (dkz) * toAdd


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

        self._central_weight = self._sampling_weights[self._matrix_size //
                                                      2, self._matrix_size //
                                                      2, self._matrix_size //
                                                      2]

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
        registration_method.SetInitialTransform(
            initial_transform, inPlace=False)

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

    moving_sitk_image_resampled = sitk.Resample(
        moving_sitk_image, fixed_sitk_image, final_transform, sitk.sitkLinear,
        0.0, moving_sitk_image.GetPixelID())

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

    N["gpts"] = np.fromfile(
        fname, dtype=np.dtype('>u2'), offset=offset, count=1)[0]
    offset += N["gpts"].size * N["gpts"].itemsize

    N["groups"] = np.fromfile(
        fname, dtype=np.dtype('>u2'), offset=offset, count=1)[0]
    offset += N["groups"].size * N["groups"].itemsize

    N["intl"] = np.fromfile(
        fname, dtype=np.dtype('>u2'), offset=offset, count=N["groups"])
    offset += N["intl"].size * N["intl"].itemsize

    N["params"] = np.fromfile(
        fname,
        dtype=np.dtype('>u2'),
        offset=256 + 4 + N["groups"] * 2,
        count=1)[0]
    offset += N["params"].size * N["params"].itemsize

    params = np.fromfile(
        fname, dtype=np.dtype('>f8'), offset=offset, count=N["params"])
    offset += params.size * params.itemsize

    wave = np.fromfile(fname, dtype=np.dtype('>i2'), offset=offset)
    offset += wave.size * wave.itemsize

    grad = np.swapaxes(
        wave.reshape((N["groups"], N["intl"][0], N["gpts"])), 0, 2)

    # set stop bit to 0
    grad[-1, ...] = 0

    # scale gradients to SI units (T/m)
    grad = (grad / 32767) * (params[3] / 100)

    # bandwidth in (Hz)
    bw = 1e6 / params[7]

    # (proton) field of view in (m)
    fov = params[1] / 100

    return grad, bw, fov, desc, N, params


def tpi_t2biexp_fft(
        image: cp.ndarray,
        T2map_short: cp.ndarray,
        T2map_long: cp.ndarray,
        acquired_grid_n: int,
        echo_time_ms: float,
        num_time_bins: int,
        shorter_readout_factor: float = 1.) -> [cp.ndarray, cp.ndarray]:
    """Forward TPI operator using FFT approximation, assuming bi-exponential T2* decay

        Parameters
        ----------
        image : input image
        T2map_short: image of short T2* component
        T2map_long: image of long T2* component
        acquired_grid_n: matrix size, assuming cubic matrix
        echo_time_ms : float
        num_time_bins: int, number of time bins
        shorter_readout_factor : shorter readout time wrt the hard coded TPI with max grad=0.16G/cm

        Returns
        ----------
        data : k-space data as would be approximately acquired with TPI
        k_mask : spherical mask for actually acquired k-space samples

        """

    ishape = image.shape
    # hard coded equation for TPI readout time
    # as a function of the distance from k-space center
    pad_factor = ishape[0] // acquired_grid_n
    k = RadialKSpacePartitioner(ishape, num_time_bins, pad_factor)

    time_bins = tpi_readout_time_from_k(k.k) / shorter_readout_factor
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


def tpi_readout_time_from_k(k):
    """Mapping of k-space vector magnitude to readout time (ms) for TPI sequence
       with hard-coded fixed parameters, max readout gradient = 0.16 G/cm
    """

    eta: float = 0.9830
    c1: float = 0.54
    c2: float = 0.46
    alpha_sw_tpi: float = 18.95
    beta_sw_tpi: float = -0.5171
    t0_sw: float = 0.0018

    # the point until the readout is linear
    m = 1126 * 0.16

    k_lin = t0_sw * m

    i1 = np.where(k <= k_lin)
    i2 = np.where(k > k_lin)

    t = np.zeros(k.shape)

    t[i1] = k[i1] / m
    t[i2] = t0_sw + ((c2 * (k[i2]**3) - (
        (c1 / eta) * np.exp(-eta * (k[i2]**3))) - beta_sw_tpi) /
                     (3 * alpha_sw_tpi))

    # convert to ms
    t *= 1000

    return t


class RadialKSpacePartitioner:
    """Partition Cartesion volume of kspace points into a number of equidistant shells

        Parameters
        ----------

        data_shape : k-space dimensions

        pad_factor : ratio of maximum spatial frequency in the output k-space and k_edge
                     the frequencies larger than k_edge are set to zero

        n_bins : number of equidistant shells

        k_edge : real maximum spatial frequency reached by the pulse sequence


        Returns
        -------
        Stores the indices of k-space points for each shell in a (possibly padded) k-space, sampling mask,
        k-space vector magnitudes for each shell
    """

    def __init__(self,
                 data_shape: Tuple,
                 n_bins: int,
                 pad_factor: float = 1.,
                 k_edge: float = 1.8 * 0.8197) -> None:

        self._n_bins = n_bins
        self._k_edge = k_edge

        # the coordinates of k-space data voxels
        k0, k1, k2 = np.meshgrid(
            np.linspace(
                -pad_factor * k_edge,
                pad_factor * k_edge,
                data_shape[0],
                endpoint=False),
            np.linspace(
                -pad_factor * k_edge,
                pad_factor * k_edge,
                data_shape[1],
                endpoint=False),
            np.linspace(
                -pad_factor * k_edge,
                pad_factor * k_edge,
                data_shape[2],
                endpoint=False))

        # k-space vector magnitude for each k-space voxel
        abs_k = np.sqrt(k0**2 + k1**2 + k2**2)
        abs_k = np.fft.fftshift(abs_k)

        # shells edges
        k_1d = np.linspace(0, k_edge, n_bins + 1)

        # k-space sample indices per shell
        self._k_inds = []
        # sampling mask

        self._kmask = np.zeros(data_shape, dtype=np.uint8)
        # k vector magnitude per shell
        self._k = np.zeros(n_bins)

        for i in range(n_bins):
            rinds = np.where(
                np.logical_and(abs_k >= k_1d[i], abs_k < k_1d[i + 1]))
            self._k_inds.append(rinds)
            self.kmask[rinds] = 1
            self._k[i] = 0.5 * (k_1d[i] + k_1d[i + 1])

        # convert mutable list to tuple
        self._k_inds = tuple(self._k_inds)

    @property
    def kmask(self) -> np.ndarray:
        return self._kmask

    @property
    def k(self) -> np.ndarray:
        return self._k

    @property
    def k_edge(self) -> float:
        return self._k_edge

    @property
    def n_bins(self) -> int:
        return self._n_bins

    @property
    def k_inds(self) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return self._k_inds


def ideal_observer_statistic(data: np.ndarray, expect_normal: np.ndarray,
                             expect_pathological: np.ndarray,
                             noise_var: float):
    """Ideal observer statistic for known background/foreground task
    Assumption: multivariate Gaussian data with constant diagonal covariance matrix (iid Gaussian noise)

    Parameters
    ----------

    data : np.ndarray
        measured noisy data
    expect_normal : np.ndarray
        data expectation for normal subject
    expect_pathological : np.ndarray
        data expectation for pathological subject
    noise_var : float
        noise variance (scalar), idd Gaussian noise

    Returns
    -------

    ideal observer statistic (likelihood ratio between two hypotheses) : scalar
    """

    expect_normal = np.hstack(
        [expect_normal.ravel().real,
         expect_normal.ravel().imag])
    expect_pathological = np.hstack(
        [expect_pathological.ravel().real,
         expect_pathological.ravel().imag])
    expect_task = expect_pathological - expect_normal

    data = np.hstack([data.ravel().real, data.ravel().imag])

    # simplified formula for iid noise
    lk_ratio = expect_task * data / noise_var
    lk_ratio = np.sum(lk_ratio)

    return lk_ratio


def ideal_observer_snr(expect_normal: np.ndarray,
                       expect_pathological: np.ndarray, noise_var: float):
    """Overall SNR of multivariate Gaussian data with constant diagonal covariance matrix (iid Gaussian noise)
    SNR of ideal observer for known background/foreground and noise cov

    Parameters
    ----------

    expect_normal : np.ndarray
        data expectation for normal subject
    expect_pathological : np.ndarray
        data expectation for pathological subject
    noise_var : float
        noise variance (scalar), idd Gaussian noise

    Returns
    -------

    Ideal observer SNR (squared version) : scalar
    """

    # as data may be complex, unravel real and imaginary components
    # real and imaginary components are assumed to have the same iid Gaussian
    # noise
    expect_normal = np.hstack(
        [expect_normal.ravel().real,
         expect_normal.ravel().imag])
    expect_pathological = np.hstack(
        [expect_pathological.ravel().real,
         expect_pathological.ravel().imag])

    # expected pathological features
    expect_task = expect_pathological - expect_normal

    # simplified formula for iid noise
    snr_square = expect_task**2 / noise_var

    snr = np.sqrt(np.sum(snr_square))

    return snr


def real_to_complex(z):
    """Convert 1D array with concatenated real datatype representation into complex datatype

       Parameters
       ----------
       z : real 1D numpy.ndarray
           concatenated real and imaginary components

       Returns
       ----------
       complex 1D numpy.ndarray
    """

    return z[:len(z) // 2] + 1j * z[len(z) // 2:]


def complex_to_real(z):
    """Convert 1D array with complex datatype into concatenated real datatype representation

       Parameters
       ----------
       z : complex 1D numpy.ndarray

       Returns
       ----------
       real 1D numpy.ndarray
           concatenated real and imaginary components
    """

    return np.concatenate((np.real(z), np.imag(z)))

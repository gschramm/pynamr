"""script to regrid 3D TPI (twisted projection) k-space data"""

import argparse
import json
import numpy as np
from pathlib import Path

import pynufft

import pymirc.viewer as pv
from pymirc.image_operations import zoom3d

from utils import setup_blob_phantom, setup_brainweb_phantom, read_tpi_gradient_files, TriliniearKSpaceRegridder, tpi_sampling_density

if __name__ == '__main__':

    #---------------------------------------------------------------------
    # input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--gradient_strength',
                        type=int,
                        default=16,
                        choices=[16, 24, 32, 48])
    parser.add_argument('--no_decay', action='store_true')
    parser.add_argument('--jitter_truth', action='store_true')
    parser.add_argument('--phantom',
                        choices=['brainweb', 'blob'],
                        default='brainweb')
    args = parser.parse_args()

    # read the data root directory from the config file
    with open('.simulation_config.json', 'r') as f:
        data_root_dir: str = json.load(f)['data_root_dir']

    gradient_strength = args.gradient_strength
    no_decay = args.no_decay
    phantom = args.phantom
    jitter_truth = args.jitter_truth

    # jitter truth suffix
    jitter_suffix = '_jit1' if jitter_truth else ''

    simulation_matrix_size: int = 256
    field_of_view_cm: float = 22.
    phantom_data_path: Path = Path(data_root_dir) / 'brainweb54'

    if gradient_strength == 16:
        gradient_file: str = str(
            Path(data_root_dir) / 'tpi_gradients/n28p4dt10g16_23Na_v1')
    elif gradient_strength == 24:
        gradient_file: str = str(
            Path(data_root_dir) / 'tpi_gradients/n28p4dt10g24f23')
    elif gradient_strength == 32:
        gradient_file: str = str(
            Path(data_root_dir) / 'tpi_gradients/n28p4dt10g32f23')
    elif gradient_strength == 48:
        gradient_file: str = str(
            Path(data_root_dir) / 'tpi_gradients/n28p4dt10g48f23')
    else:
        raise ValueError

    # the number of time steps for the data simulation
    num_time_bins: int = 300

    # the two echo times in ms
    t_echo_1_ms: float = 0.5
    t_echo_2_ms: float = 5.

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
        if jitter_truth:
            T2long_ms_csf: float = 51.
            T2long_ms_gm: float = 18.
            T2long_ms_wm: float = 20.
            T2short_ms_csf: float = 51.
            T2short_ms_gm: float = 5.
            T2short_ms_wm: float = 6.
        else:
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
        if jitter_truth:
            na_image, t1_image, T2short_ms, T2long_ms = setup_brainweb_phantom(
            simulation_matrix_size,
            phantom_data_path,
            field_of_view_cm=field_of_view_cm,
            csf_na_concentration=3.,
            gm_na_concentration=1.4,
            wm_na_concentration=1.1,
            T2long_ms_csf=T2long_ms_csf,
            T2long_ms_gm=T2long_ms_gm,
            T2long_ms_wm=T2long_ms_wm,
            T2short_ms_csf=T2short_ms_csf,
            T2short_ms_gm=T2short_ms_gm,
            T2short_ms_wm=T2short_ms_wm)
        else:
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
        data_root_dir) / f'{phantom}_{Path(gradient_file).name}{decay_suffix}{jitter_suffix}'
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

    gridded_data_matrix_size: int = 128
    delta_k = 1 / field_of_view_cm

    # calculate vector of TPI sampling densities for all kspace points in the non-uniform data
    sampling_density = tpi_sampling_density(k0.ravel(), k1.ravel(), k2.ravel(),
                                            kp)

    regridder = TriliniearKSpaceRegridder(gridded_data_matrix_size, delta_k,
                                          k0.ravel(), k1.ravel(), k2.ravel(),
                                          sampling_density, kmax)

    regridded_data_echo_1 = regridder.regrid(nonuniform_data_long_echo_1)

    nufft_norm = 11626.0
    regridded_data_echo_1 /= nufft_norm

    ifft_echo_1 = np.fft.ifftn(regridded_data_echo_1, norm='ortho')

    # interpolate magnitude images to simulation grid size (which can be different from gridded data size)
    a = zoom3d(np.abs(ifft_echo_1),
               simulation_matrix_size / gridded_data_matrix_size)

    vi = pv.ThreeAxisViewer([na_image, a])

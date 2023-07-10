""" Ideal observer study for comparing different max gradients for TPI:
    more noise vs less decay tradeoff

    Phantoms:
    - brainweb
    - cylinder

    Data simulation implementation using:
    - NUFFT and real gradient files
    - approximative FFT (does not take into account the lower noise in the radial center)

    The code could be nicer and more general but large matrix sizes for the simulation
    require some tweaks for avoiding GPU RAM overflow
"""

import argparse
import h5py
import json
from pathlib import Path
from copy import deepcopy
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndimage
import sigpy
import sys
import time
from pymirc.image_operations import zoom3d
import pymirc.viewer as pv

from utils import setup_blob_phantom, setup_brainweb_phantom, read_tpi_gradient_files, crop_kspace_data, simpleForward_TPI_FFT
from utils_sigpy import NUFFTT2starDualEchoModel, recon_empirical_grid_and_ifft, recon_gridding, recon_tpi_iterative_nufft, NUFFT_TPI_BiexpModel

from scipy.ndimage import binary_erosion
from scipy.interpolate import interp1d

from matplotlib import pyplot as plt


def simulate_data_ideal_observer_fft(x_normal: cp.ndarray,
                      x_patho: cp.ndarray,
                      T2short_ms: cp.ndarray,
                      T2long_ms: cp.ndarray,
                      grid_n: int,
                      acc: float,
                      echo_time_1_ms: int,
                      num_time_bins: int,
                      noise_level: float):
        """Helper function for simulating expected and noisy data for a single TPI acquisition
            using approximative simple FFT, for the normal and the pathological image
        """

        # simple single echo cupy fft implementation
        data_x_normal, k_mask = simpleForward_TPI_FFT(x_normal, T2short_ms, T2long_ms, grid_n,
                echo_time_1_ms, num_time_bins, acc)
        data_x_patho, k_mask = simpleForward_TPI_FFT(x_patho, T2short_ms, T2long_ms, grid_n,
                echo_time_1_ms, num_time_bins, acc)

        data_expect_x_normal = data_x_normal.copy()
        data_expect_x_patho = data_x_patho.copy()


        #--------------------------------------------------------------------------
        # add noise to the data

        # noise level given wrt to the max
        nl = noise_level * cp.abs(data_x_normal.max())
        # add approximative noise increase due to fewer samples for higher gradients
        # have to simulate this by hand and not entirely correct,
        # because central radial part has a varying lower noise level
        nl *= np.sqrt(acc)

        # must confine the noise solely to acquired samples (apply k space mask)
        data_x_normal += nl * (cp.random.randn(*data_x_normal.shape) +
                     1j * cp.random.randn(*data_x_normal.shape)) *  k_mask
        data_x_patho += nl * (cp.random.randn(*data_x_patho.shape) +
                     1j * cp.random.randn(*data_x_patho.shape)) *  k_mask

        return data_expect_x_normal, data_x_normal, data_expect_x_patho, data_x_patho, nl



def simulate_data_ideal_observer_nufft(x_normal: cp.ndarray,
                              x_patho: cp.ndarray,
                              data_root_dir: str,
                              gradient_strength: int,
                              field_of_view_cm: float,
                              acq_sampling_time_ms: float,
                              time_bin_width_ms: int,
                              echo_time_1_ms: float,
                              T2short_ms: np.ndarray,
                              T2long_ms: np.ndarray,
                              noise_level: float,
                              const_readout_time: bool):
    """ Helper function for simulating expected and noisy data for a single SW-TPI acquisition
        using NUFFT and a gradient trace file, for the normal and the pathological image
    """
    # find trajectory gradient file
    if gradient_strength == 16 or gradient_strength == 8:
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
    if gradient_strength == 8:
        num_time_pts = kx.shape[1]
        interp_time = interp1d(np.arange(num_time_pts), kx, axis=1)
        oversample2x = np.arange(0, num_time_pts-0.9, 0.5)
        kx = interp_time(oversample2x)
        interp_time = interp1d(np.arange(num_time_pts), ky, axis=1)
        ky = interp_time(oversample2x)
        interp_time = interp1d(np.arange(num_time_pts), kz, axis=1)
        kz = interp_time(oversample2x)
    # artificial gradient strength 64 based on 2x undersampling in time of the gradient 32 trace
    # because no gradient 64 trace available yet
    elif gradient_strength == 64:
        kx = kx[:,::2]
        ky = ky[:,::2]
        kz = kz[:,::2]

    # group k-space coordinates 
    k_1_cm = np.stack([kx,ky,kz], axis=-1)
    ## reshape as (num_time_samples, num_readouts, space_dim)
    k_1_cm = np.transpose(k_1_cm, (1,0,2))

    # readout time kept constant with respect to grad 16
    # by going back and forth along traj
    if const_readout_time:
        if gradient_strength == 32:
            k_1_cm = np.concatenate([k_1_cm, k_1_cm[::-1,:,:]], axis=0)
        elif gradient_strength == 48:
            k_1_cm = np.concatenate([k_1_cm, k_1_cm[::-1,:,:], k_1_cm], axis=0)
        elif gradient_strength == 64:
            k_1_cm = np.concatenate([k_1_cm, k_1_cm[::-1,:,:], k_1_cm, k_1_cm[::-1,:,:]], axis=0)

    # the gradient files only contain a half sphere
    k_1_cm = np.concatenate([k_1_cm, -k_1_cm], axis=1)

# different gradient files, not implemented
# read gradients in T/m with shape (num_samples_per_readout, num_readouts, 3)

#with h5py.File(Path(data_root_dir) / 'tpi_gradients/ak_grad56.h5',
#               'r') as data:
#    grads_T_m = np.transpose(data['/gradients'][:], (2, 1, 0))

    # different grad files, TODO properly
#    grads_T_m = np.load(Path(data_root_dir) / 'tpi_gradients/ak_grad56.npy')
#    acq_sampling_time_ms = 0.016
#    dt_us = acq_sampling_time_ms * 1000
#    # k values in 1/cm with shape (num_samples_per_readout, num_readouts, 3)
#    k_1_cm = 0.01 * np.cumsum(grads_T_m, axis=0) * dt_us * gamma_by_2pi_MHz_T
#    # undersample for simulating different gradients
#    k_1_cm = k_1_cm[::g+1,:,:]


    k_1_cm_abs = np.linalg.norm(k_1_cm, axis=2)
    print(f'readout kmax .: {k_1_cm_abs.max():.2f} 1/cm')
    print(f'64 kmax      .: {kmax64_1_cm:.2f} 1/cm')

    num_sim_chunks = 1
    data_x_normal = []
    data_x_patho = []

    # split nufft into chunks to avoid GPU RAM overflow
    for i_chunk, k_inds in enumerate(
                np.array_split(np.arange(k_1_cm.shape[0]), num_sim_chunks)):

        data_model = NUFFT_TPI_BiexpModel(sim_shape,
                         k_1_cm[k_inds,...],
                         T2short_ms,
                         T2long_ms,
                         field_of_view_cm,
                         acq_sampling_time_ms,
                         time_bin_width_ms,
                         echo_time_1_ms + k_inds[0] * acq_sampling_time_ms)

        data_operator = data_model.get_operator_w_decay_model()
 
        data_x_normal.append(data_operator(x_normal))
        data_x_patho.append(data_operator(x_patho))
        del data_operator

    # concatenate all data chunks
    data_x_normal = cp.concatenate(data_x_normal)
    data_x_patho = cp.concatenate(data_x_patho)

    # save noise-free data as expectation
    data_expect_x_normal = data_x_normal.copy()
    data_expect_x_patho = data_x_patho.copy()

    # add noise to the data
    nl = noise_level * cp.abs(data_x_normal.max())
    data_x_normal += nl * (cp.random.randn(*data_x_normal.shape) +
                     1j * cp.random.randn(*data_x_normal.shape))
    data_x_patho += nl * (cp.random.randn(*data_x_patho.shape) +
                     1j * cp.random.randn(*data_x_patho.shape))

    return data_expect_x_normal, data_x_normal, data_expect_x_patho, data_x_patho, nl, k_1_cm


def ideal_observer_snr(expect_normal: np.ndarray, expect_pathological: np.ndarray, noise_cov: float):
    """
    Overall SNR of multivariate Gaussian data with constant diagonal covariance matrix (iid Gaussian noise)
    SNR of ideal observer for known background/foreground and noise cov

    Parameters
    ----------
    expect_normal : ndarray of data expectation for normal subject
    expect_pathological : ndarray of data expectation for pathological subject
    noise_cov : noise covariance (scalar), idd Gaussian noise

    Returns
    -------
    SNR : scalar
    """

    # as data may be complex, unravel real and imaginary components
    # real and imaginary components are assumed to have the same iid Gaussian noise
    expect_normal = np.hstack([expect_normal.ravel().real, expect_normal.ravel().imag])
    expect_pathological = np.hstack([expect_pathological.ravel().real, expect_pathological.ravel().imag])

    # expected pathological features
    expect_task = expect_pathological - expect_normal

    # simplified formula for iid noise
    snr_square = expect_task**2 / noise_cov

    snr = np.sqrt(np.sum(snr_square))

    return snr


def ideal_observer_statistic(data: np.ndarray, expect_normal: np.ndarray, expect_pathological: np.ndarray, noise_cov: float):
    """
    Ideal observer statistic for known background/foreground task
    Assumption: multivariate Gaussian data with constant diagonal covariance matrix (iid Gaussian noise)

    Parameters
    ----------
    data : measured noisy data
    expect_normal : ndarray of data expectation for normal subject
    expect_pathological : ndarray of data expectation for pathological subject
    noise_cov : noise covariance (scalar), idd Gaussian noise

    Returns
    -------
    ideal observer statistic (likelihood ratio between two hypotheses) : scalar
    """

    expect_normal = np.hstack([expect_normal.ravel().real, expect_normal.ravel().imag])
    expect_pathological = np.hstack([expect_pathological.ravel().real, expect_pathological.ravel().imag])
    expect_task = expect_pathological - expect_normal

    data = np.hstack([data.ravel().real, data.ravel().imag])

    # simplified formula for iid noise
    lk_ratio = expect_task * data / noise_cov
    lk_ratio = np.sum(lk_ratio)

    return lk_ratio


#--------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--noise_level', type=float, default=1e-2, help='noise level relative to DC/max component')
parser.add_argument('--change_perc', type=float, default=10, help='pathological change in percentage of normal tissue')
parser.add_argument('--ft', type=str, default='nufft')
#parser.add_argument('--gradient_strengths', type=int, default=[8, 16, 32, 48]) # [e-2 G/cm]
parser.add_argument('--phantom',
                    type=str,
                    default='brainweb',
                    choices=['brainweb', 'blob'])
parser.add_argument('--pathology',
                    type=str,
                    default='none',
                    help='relevant combinations of: lesion, big, small, gm, wm, gwm, cos')
parser.add_argument('--nodecay', action='store_true')
parser.add_argument('--padFOV', action='store_true', help='either for smaller sim deltak or for avoiding aliasing at the edges with nufft')
parser.add_argument('--sigma', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--const_readout_time', action='store_true', help='keep the readout time constant by going back and forth along trajectories')
#parser.add_argument('--odir_suffix', type=str, default='')
args = parser.parse_args()

noise_level = args.noise_level
#gradient_strengths = args.gradient_strengths
phantom = args.phantom
nodecay = args.nodecay
sigma = args.sigma
seed = args.seed
#odir_suffix = args.odir_suffix
change_perc = args.change_perc
pathology = args.pathology
ft = args.ft
padFOV = args.padFOV
const_readout_time = args.const_readout_time

cp.random.seed(seed)
plt.ion()

#---------------------------------------------------------------
# fixed parameters

gradient_strengths = [16, 32, 48, 64] #[16, 32, 48]

# matrix size: assumed cubic
sim_shape = (256, 256, 256)
grid_shape = (64, 64, 64)

# FOV
field_of_view_cm: float = 22.

# 10us sampling time
acq_sampling_time_ms: float = 0.01 #0.016

# time sampling step in micro seconds
dt_us = acq_sampling_time_ms * 1000
# gamma by 2pi in MHz/T
gamma_by_2pi_MHz_T: float = 11.262

# kmax for 64 matrix (edge)
kmax64_1_cm = 1 / (2 * field_of_view_cm / 64)

# echo times in ms
echo_time_1_ms = 0.5 #0.455
echo_time_2_ms = 5.

# signal fraction decaying with short T2* time
short_fraction = 0.6

time_bin_width_ms: float = 0.25

# read the data root directory from the config file
with open('.simulation_config.json', 'r') as f:
    data_root_dir: str = json.load(f)['data_root_dir']

#odir = Path(
#    data_root_dir
#) / 'ideal_observer_brainweb' / f'{phantom}_nodecay_{nodecay}_nl_{noise_level:.1E}_s_{seed:03}{odir_suffix}'
#odir.mkdir(exist_ok=True, parents=True)

#with open(odir / 'config.json', 'w') as f:
#    json.dump(vars(args), f, indent=4)

#--------------------------------------------------------------------------
# setup the phantom images: normal, pathological

if nodecay:
    T2long_ms_csf: float = 1e7 
    T2long_ms_gm: float = 1e7
    T2long_ms_wm: float = 1e7
    T2short_ms_csf: float = 1e7
    T2short_ms_gm: float = 1e7
    T2short_ms_wm: float = 1e7
else:
    T2long_ms_csf: float = 50.
    T2long_ms_gm: float = 18.
    T2long_ms_wm: float = 20.
    T2short_ms_csf: float = 50.
    T2short_ms_gm: float = 3.
    T2short_ms_wm: float = 3.

phantom_data_path: Path = Path(data_root_dir) / 'brainweb54'

# setup the brainweb phantom with the given simulation matrix size
if phantom == 'brainweb':
    # normal image
    x_normal, t1_image, T2short_ms, T2long_ms = setup_brainweb_phantom(
        sim_shape[0],
        phantom_data_path,
        field_of_view_cm=field_of_view_cm,
        T2long_ms_csf=T2long_ms_csf,
        T2long_ms_gm=T2long_ms_gm,
        T2long_ms_wm=T2long_ms_wm,
        T2short_ms_csf=T2short_ms_csf,
        T2short_ms_gm=T2short_ms_gm,
        T2short_ms_wm=T2short_ms_wm,
        csf_na_concentration=1.5,
        gm_na_concentration=0.7,
        wm_na_concentration=0.5,
        other_na_concentration=0.5,
        add_anatomical_mismatch=False,
        add_T2star_bias=False)

    # pathological image
    x_patho, t1_image, T2short_ms, T2long_ms = setup_brainweb_phantom(
        sim_shape[0],
        phantom_data_path,
        field_of_view_cm=field_of_view_cm,
        T2long_ms_csf=T2long_ms_csf,
        T2long_ms_gm=T2long_ms_gm,
        T2long_ms_wm=T2long_ms_wm,
        T2short_ms_csf=T2short_ms_csf,
        T2short_ms_gm=T2short_ms_gm,
        T2short_ms_wm=T2short_ms_wm,
        csf_na_concentration=1.5,
        gm_na_concentration=0.7,
        wm_na_concentration=0.5,
        other_na_concentration=0.5,
        add_anatomical_mismatch=False,
        add_T2star_bias=False,
        pathology=pathology,
        pathology_change_perc=change_perc
        )

    del t1_image

# simple cylinder
elif phantom == 'blob':
    # normal image
    x_normal, t1_image, T2short_ms, T2long_ms = setup_blob_phantom(sim_shape[0],
                                                            radius=0.65,
                                                            T2short=3., T2long=18.,
                                                            longerT2ring=False)
    # pathological image
    x_patho, t1_image, T2short_ms, T2long_ms = setup_blob_phantom(sim_shape[0],
                                                            radius=0.65,
                                                            T2short=3., T2long=18.,
                                                            longerT2ring=False,
                                                            pathology=pathology,
                                                            pathology_change_perc=change_perc)
    del t1_image
else:
    raise ValueError

# pad the phantom to a larger FOV
# k_max stays the same, delta_k becomes smaller
if padFOV:
    pad_factor = 1.3 #2
    padded_shape = tuple(int(round(x * pad_factor)) for x in sim_shape)
    padding_len =  (padded_shape[0] - sim_shape[0]) // 2
    x_normal = np.pad(x_normal, padding_len)
    x_patho = np.pad(x_patho, padding_len)
    # the padding value shouldn't matter because pathology only in brain, zero elsewhere
    T2short_ms = np.pad(T2short_ms, padding_len, constant_values=0.5*np.finfo(np.float32).min)
    T2long_ms = np.pad(T2long_ms, padding_len, constant_values=0.5*np.finfo(np.float32).min)

    # update relevant parameters, not great
    sim_shape = padded_shape
    field_of_view_cm *= pad_factor
    grid_shape = tuple(int(round(x * pad_factor)) for x in grid_shape)

# move image to GPU
x_normal = cp.asarray(x_normal.astype(np.complex128))
x_patho = cp.asarray(x_patho.astype(np.complex128))

# min/max values for display
data_max = cp.max(cp.abs(cp.fft.fftn(x_normal, norm='ortho')))
im_max =  cp.max(cp.abs(x_patho))
im_min =  cp.min(cp.abs(x_patho))

# must save handles of pymirc viewer figures to keep them alive and responsive
v = []

# show initial images and pathology in image space
im1 = cp.asnumpy(cp.abs(x_normal))
im2 = cp.asnumpy(cp.abs(x_patho))
im3 = cp.asnumpy(cp.abs(x_normal - x_patho))
v.append(pv.ThreeAxisViewer([im1, im2, im3], imshow_kwargs=[{'cmap':'viridis', 'vmax':im_max, 'vmin':im_min}, {'cmap':'viridis', 'vmax':im_max, 'vmin':im_min}, {'cmap':'viridis', 'vmax':im3.max(), 'vmin':im3.min()}]))
#plt.savefig(f'/users/nexuz2/mfilip0/{phantom}_{pathology}_images.png')


#--------------------------------------------------------------------------
# simulate data

# init vars for ideal observer analysis
snr = cp.zeros(len(gradient_strengths), dtype=float)
statistic = cp.zeros(len(gradient_strengths), dtype=float)

# for checking and reconstruction 
data_x_patho_for_check = []
data_expect_x_patho_for_check = []
data_x_normal_for_check = []
data_expect_x_normal_for_check = []
k_1_cm_for_check = []

# loop over pulse seq setups/trajectories with different max gradients
for g, gradient_strength in enumerate(gradient_strengths):

    if ft == 'nufft':
        # simulate expected and noisy data for the first echo using the NUFFT dual echo model
        data_expect_x_normal, data_x_normal, data_expect_x_patho, data_x_patho, nl, k_1_cm = simulate_data_ideal_observer_nufft(
                                                x_normal,
                                                x_patho,
                                                data_root_dir,
                                                gradient_strength,
                                                field_of_view_cm,
                                                acq_sampling_time_ms,
                                                time_bin_width_ms,
                                                echo_time_1_ms,
                                                T2short_ms,
                                                T2long_ms,
                                                noise_level,
                                                const_readout_time)
        k_1_cm_for_check.append(k_1_cm)

    elif ft == 'fft':
        # simulate expected and noisy data for the first echo using simple FFT
        data_expect_x_normal, data_x_normal, data_expect_x_patho, data_x_patho, nl = simulate_data_ideal_observer_fft(
                                                        x_normal,
                                                        x_patho,
                                                        cp.asarray(T2short_ms),
                                                        cp.asarray(T2long_ms),
                                                        grid_shape[0],
                                                        gradient_strengths[g] / gradient_strengths[0],
                                                        echo_time_1_ms,
                                                        grid_shape[0]//2,
                                                        noise_level)

    # for visualizing and checking
    data_x_patho_for_check.append(data_x_patho)
    data_expect_x_patho_for_check.append(data_expect_x_patho)
    data_x_normal_for_check.append(data_x_normal)
    data_expect_x_normal_for_check.append(data_expect_x_normal)

    # Compute ideal observer SNR and statistic
    snr[g] = ideal_observer_snr(data_expect_x_normal, data_expect_x_patho, nl**2)
    statistic[g] = ideal_observer_statistic(data_x_patho, data_expect_x_normal, data_expect_x_patho, nl**2)

# print ideal observer SNR and statistic
np.set_printoptions(precision=2) #, floatmode='unique')
print(f'\n snr for grad 16: {snr[0]:.2f}')
print(f' snr relative to first grad {snr/snr[0]}')
print(f' all snrs {snr}')
print(f'\n statistic for patho for grad 16: {statistic[0]:.2f} ')
print(f' statistic relative to first grad {statistic/statistic[0]}')
print(f' all statistics for patho {statistic}')

#sys.exit()
#--------------------------------------------------------------------------
# Visualize data for the last max gradient, mostly for checking

if ft == 'nufft':
    # flat data sorted according to time bins
    for g, grad in enumerate(gradient_strengths):
        data_diff_expect = data_expect_x_normal_for_check[g] - data_expect_x_patho_for_check[g]
        data_diff = data_x_normal_for_check[g] - data_x_patho_for_check[g]
        data_show = cp.asnumpy(cp.abs(data_diff_expect))
        plt.figure(), plt.plot(data_show), plt.suptitle(f'Expected pathology in nufft data space grad {grad}')
        #plt.savefig(f'/users/nexuz2/mfilip0/{phantom}_expected_{pathology}_nufft_data.pdf')
        data_show = cp.asnumpy(cp.abs(data_diff))
        plt.figure(), plt.plot(data_show), plt.suptitle(f'Pathology in nufft data space grad {grad}')
        #plt.savefig(f'/users/nexuz2/mfilip0/{phantom}_{pathology}_nufft_data.pdf')
elif ft=='fft':
    # show only the last gradient
    # easier to visualize
    data_diff_expect = data_expect_x_normal - data_expect_x_patho
    data_diff = data_x_normal - data_x_patho
    # show log data normal expect and noisy
    im1 = cp.asnumpy(cp.abs(cp.fft.fftshift(data_expect_x_normal)))
    im2 =  cp.asnumpy(cp.abs(cp.fft.fftshift(cp.fft.fftn(x_normal))))
    v.append(pv.ThreeAxisViewer([np.log(im1, where=im1>0.), np.log(im2, where=im2>0.)], imshow_kwargs={'cmap':'viridis','vmax':np.log(data_max)}))

    # show log data patho expect and noisy
    im1 = cp.asnumpy(cp.abs(cp.fft.fftshift(data_expect_x_patho)))
    im2 =  cp.asnumpy(cp.abs(cp.fft.fftshift(cp.fft.fftn(x_patho))))
    v.append(pv.ThreeAxisViewer([np.log(im1, where=im1>0.), np.log(im2, where=im2>0.)], imshow_kwargs={'cmap':'viridis','vmax':np.log(data_max)}))

    # show log data diff expect and noisy
    im1 = cp.asnumpy(cp.abs(cp.fft.fftshift(data_diff_expect)))
    im2 =  cp.asnumpy(cp.abs(cp.fft.fftshift(data_diff)))
    v.append(pv.ThreeAxisViewer([np.log(im1, where=im1>0.), np.log(im2, where=im2>0.)], imshow_kwargs={'cmap':'viridis'}))
    plt.savefig(f'/users/nexuz2/mfilip0/{phantom}_expected_noisy_{pathology}_log_fft_data.pdf')

    # show data diff expect and noisy
    im1 = cp.asnumpy(cp.abs(cp.fft.fftshift(data_diff_expect)))
    im2 =  cp.asnumpy(cp.abs(cp.fft.fftshift(data_diff)))
    v.append(pv.ThreeAxisViewer([im1, im2], imshow_kwargs={'cmap':'viridis'}))

#--------------------------------------------------------------------
# Visualize simply reconstructed images

if ft == 'fft':
    # reconstruct normal, patho, diff images for the last max gradient using simple ifft
    op_ifft = sigpy.linop.IFFT(grid_shape, center=False)

    # crop data to nominal resolution
    temp = crop_kspace_data(data_x_normal, sim_shape, grid_shape)
    ifft_normal = op_ifft(temp)

    # crop data to nominal resolution
    temp = crop_kspace_data(data_x_patho, sim_shape, grid_shape)
    ifft_patho = op_ifft(temp)

    # crop data to nominal resolution
    temp = crop_kspace_data(data_diff, sim_shape, grid_shape)
    ifft_diff = op_ifft(temp)

    # show reconstructed images for the last max gradient
    im1 = cp.asnumpy(cp.abs(ifft_normal))
    im2 = cp.asnumpy(cp.abs(ifft_patho))
    im3 = cp.asnumpy(cp.abs(ifft_diff))
    v.append(pv.ThreeAxisViewer([im1, im2, im3], imshow_kwargs={'cmap':'viridis', 'vmax':im_max, 'vmin':im_min}))

    # reconstruct pathological images for all the max gradients 
    ifft_expect_patho = []
    ifft_patho = []
    for g in np.arange(len(gradient_strengths)):

        # crop data to nominal resolution
        temp = crop_kspace_data(data_expect_x_patho_for_check[g], sim_shape, grid_shape)
        ifft_expect_patho.append(op_ifft(temp))

        # crop data to nominal resolution
        temp = crop_kspace_data(data_x_patho_for_check[g], sim_shape, grid_shape)
        ifft_patho.append(op_ifft(temp))

elif ft == 'nufft':
    # reconstruct pathological images for all the max gradients 
    ifft_expect_patho = []
    ifft_patho = []
    for g in range(len(gradient_strengths)):

        # patho image simple recon
        nufft_k_coords = cp.asarray(k_1_cm_for_check[g].reshape(-1, 3)) * field_of_view_cm
        #dk = acq_sampling_time_ms * 1e-3 * 1129.2 * gradient_strengths[g] * 1e-2

        # normalization factor to account for different matrix sizes between sim and recon,
        # as nufft uses also fft with norm ortho
        norm_factor = 1.
        for d in range(len(grid_shape)):
            norm_factor *= np.sqrt(grid_shape[d] / sim_shape[d])

        # noisy patho 
        im = recon_tpi_iterative_nufft(data_x_patho_for_check[g],
                          grid_shape,
                          k_1_cm_for_check[g],
                          field_of_view_cm,
                          acq_sampling_time_ms,
                          time_bin_width_ms,
                          echo_time_1_ms)
        #im = recon_gridding(data_x_patho_for_check[g], nufft_k_coords, grid_shape, sd_corr)
        ifft_patho.append(im * norm_factor)

        # expected patho
        im = recon_tpi_iterative_nufft(data_expect_x_patho_for_check[g],
                          grid_shape,
                          k_1_cm_for_check[g],
                          field_of_view_cm,
                          acq_sampling_time_ms,
                          time_bin_width_ms,
                          echo_time_1_ms)
        #im = recon_gridding(data_expect_x_patho_for_check[g], nufft_k_coords, grid_shape, sd_corr)
        ifft_expect_patho.append(im * norm_factor)


# show expected patho images for all the gradients
im = [cp.asnumpy(cp.abs(x)) for x in ifft_expect_patho]
v.append (pv.ThreeAxisViewer([*im], imshow_kwargs={'cmap':'viridis', 'vmax':im_max}))
    
# show noisy patho images for all the gradients
im = [cp.asnumpy(cp.abs(x)) for x in ifft_patho]
v.append (pv.ThreeAxisViewer([*im], imshow_kwargs={'cmap':'viridis', 'vmax':im_max}))


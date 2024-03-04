""" Ideal observer study for comparing different max gradients for TPI:
    more noise vs less decay tradeoff

    Phantoms:
    - brainweb

    Data simulation implementation using:
    - NUFFT and real gradient files
    - approximative FFT (does not take into account the lower noise in the radial center)

    The code could be nicer and more general but large matrix sizes for the simulation
    require some tweaks for avoiding GPU RAM overflow
"""

import argparse
import json
from pathlib import Path
import numpy as np
import cupy as cp
import sigpy
import sys
import time
from scipy.ndimage import zoom, gaussian_filter
import scipy.optimize as sci_opt
from matplotlib import pyplot as plt

import pymirc.viewer as pv
from utils import setup_brainweb_phantom, ideal_observer_statistic, ideal_observer_snr, tpi_kspace_coords_1_cm_scanner, tpi_t2biexp_fft, real_to_complex, complex_to_real, crop_kspace_data
from utils_sigpy import NUFFT_T2Biexp_Model, LossL2
from functools import partial

# --------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    '--noise_level',
    type=float,
    default=1e-2,
    help='noise level relative to DC/max component')
parser.add_argument(
    '--patho_change_perc',
    type=float,
    help='pathological change in percentage of normal tissue')
parser.add_argument(
    '--patho_size_perc',
    type=float,
    help='pathology max size in percentage of FOV in 1D')
parser.add_argument('--patho_center_perc', type=str, help='pathology center')
parser.add_argument(
    '--ft', type=str, default='nufft', choices=['fft', 'nufft'])
parser.add_argument(
    '--pathology',
    type=str,
    default='none',
    help=
    'glioma or relevant combinations of: lesion, big, small, gm, wm, gwm, cos')
parser.add_argument(
    '--test',
    type=str,
    default='patho_normal',
    choices=['patho_normal', 'patho_change', 'patho_res'],
    help=
    'test for presence of pathology, or for a change in a pathology over time')
parser.add_argument('--no_decay', action='store_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument(
    '--const_readout_time',
    action='store_true',
    help=
    'keep the readout time constant by going back and forth along trajectories'
)
parser.add_argument('--no_recon', action='store_true')
parser.add_argument(
    '--num_sim_chunks',
    type=int,
    default=10,
    help='number of simulation (data) chunks for avoiding GPU memory overflow')

args = parser.parse_args()

noise_level = args.noise_level
no_decay = args.no_decay
seed = args.seed
patho_change_perc = args.patho_change_perc
patho_size_perc = args.patho_size_perc
patho_center_perc = args.patho_center_perc
pathology = args.pathology
ft = args.ft
const_readout_time = args.const_readout_time
no_recon = args.no_recon
test = args.test
num_sim_chunks = args.num_sim_chunks

# ---------------------------------------------------------------
# init
#TODO check params consistency, like patho_res test incompatible with glioma or cosine
if patho_center_perc != None:
    patho_center_perc = np.fromstring(
        patho_center_perc, dtype=np.float32, sep=',')

# ---------------------------------------------------------------
# init

cp.random.seed(seed)

plt.ion()
plt.rcParams.update({'font.size': 14})

# ---------------------------------------------------------------
# fixed parameters

gradient_strengths = [16, 32, 48]
nb_grads = len(gradient_strengths)
pulse_seq = 'tpi'

# matrix size: assumed cubic
sim_shape = (256, 256, 256)
grid_shape = (64, 64, 64)
recon_shape = grid_shape

# FOV
field_of_view_cm: float = 22.

# 10us sampling time
acq_sampling_time_ms: float = 0.01  # 0.016

# time sampling step in micro seconds
dt_us = acq_sampling_time_ms * 1000
# gamma by 2pi in MHz/T
gamma_by_2pi_MHz_T: float = 11.262

# kmax for 64 matrix (edge)
kmax64_1_cm = 1 / (2 * field_of_view_cm / 64)

# echo times in ms
echo_time_ms = 0.5  # 0.455

# signal fraction decaying with short T2* time
short_fraction = 0.6

time_bin_width_ms: float = 0.25

# must save handles of pymirc viewer figures to keep them alive and responsive
v = []

# read the data root directory from the config file
with open('.simulation_config.json', 'r') as f:
    data_root_dir: str = json.load(f)['data_root_dir']

# phantom folder
phantom_data_path: Path = Path(data_root_dir) / 'brainweb54'

# --------------------------------------------------------------------------
# setup the phantom images

# T2* values
if no_decay:
    T2long_ms_csf: float = 1e7
    T2long_ms_gm: float = 1e7
    T2long_ms_wm: float = 1e7
    T2long_ms_other: float = 1e7
    T2short_ms_csf: float = 1e7
    T2short_ms_gm: float = 1e7
    T2short_ms_wm: float = 1e7
    T2short_ms_other: float = 1e7
else:
    T2long_ms_csf: float = 50.
    T2long_ms_gm: float = 18.
    T2long_ms_wm: float = 20.
    T2long_ms_other: float = 18.
    T2short_ms_csf: float = 50.
    T2short_ms_gm: float = 3.
    T2short_ms_wm: float = 3.
    T2short_ms_other: float = 3.

csf_na_concentration = 1.5
gm_na_concentration = 0.7
wm_na_concentration = 0.5
other_na_concentration = 0.5

# shorter call to setup brainweb, handy for repeated calls with same params
short_setup_brainweb = partial(
    setup_brainweb_phantom,
    sim_shape[0],
    phantom_data_path,
    field_of_view_cm=field_of_view_cm,
    T2long_ms_csf=T2long_ms_csf,
    T2long_ms_gm=T2long_ms_gm,
    T2long_ms_wm=T2long_ms_wm,
    T2long_ms_other=T2long_ms_other,
    T2short_ms_csf=T2short_ms_csf,
    T2short_ms_gm=T2short_ms_gm,
    T2short_ms_wm=T2short_ms_wm,
    T2short_ms_other=T2short_ms_other,
    csf_na_concentration=csf_na_concentration,
    gm_na_concentration=gm_na_concentration,
    wm_na_concentration=wm_na_concentration,
    other_na_concentration=other_na_concentration)

# setup the brainweb phantom with the given simulation matrix size
if test == 'patho_normal':
    # normal image
    im_baseline, t1_image, T2short_baseline_ms, T2long_baseline_ms = short_setup_brainweb(
    )

    # pathological image
    im_w_change, t1_image, T2short_w_change_ms, T2long_w_change_ms = short_setup_brainweb(
        pathology=pathology,
        pathology_size_perc=patho_size_perc,
        pathology_change_perc=patho_change_perc,
        pathology_center_perc=patho_center_perc)

elif test == 'patho_change':
    # pathology state 1
    im_baseline, t1_image, T2short_baseline_ms, T2long_baseline_ms = short_setup_brainweb(
        pathology=pathology,
        pathology_size_perc=patho_size_perc,
        pathology_change_perc=patho_change_perc,
        pathology_center_perc=patho_center_perc)

    # pathology state 2
    im_w_change, t1_image, T2short_w_change_ms, T2long_w_change_ms = short_setup_brainweb(
        pathology=pathology,
        pathology_size_perc=patho_size_perc,
        pathology_change_perc=patho_change_perc * 0.7,
        pathology_center_perc=patho_center_perc)

elif test == 'patho_res':
    # no pathology
    im_none, t1_image, T2short_baseline_ms, T2long_baseline_ms = short_setup_brainweb(
    )

    # pathology higher res
    im_baseline, t1_image, T2short_baseline_ms, T2long_baseline_ms = short_setup_brainweb(
        pathology=pathology,
        pathology_size_perc=patho_size_perc,
        pathology_change_perc=patho_change_perc,
        pathology_center_perc=patho_center_perc)

    # pathology lower res
    im_w_change, t1_image, T2short_w_change_ms, T2long_w_change_ms = short_setup_brainweb(
        pathology=pathology,
        pathology_size_perc=patho_size_perc + 1.6,
        pathology_change_perc=patho_change_perc,
        pathology_center_perc=patho_center_perc)

    diff1 = im_baseline - im_none
    diff2 = im_w_change - im_none
    #    im_w_change = im_baseline + diff2
    #    diff2 = im_w_change - im_none
    sum2 = np.sum(diff2, where=np.abs(diff2) > 1e-3 * np.abs(diff2).max())
    sum1 = np.sum(diff1, where=np.abs(diff1) > 1e-3 * np.abs(diff1).max())
    #    diff1_norm = sum2 / sum1
    #    im_baseline = im_none + diff1 * diff1_norm
    diff2_norm = sum1 / sum2
    im_w_change = im_none + diff2 * diff2_norm
    sum_diff = np.sum(im_w_change - im_baseline)
    print(f'{sum_diff} should be around 0., scale factor = {diff2_norm}')

# orient the images so that they show in LPS in the pymirc viewer
im_baseline = im_baseline[::-1, ::-1]
im_w_change = im_w_change[::-1, ::-1]
T2short_baseline_ms = T2short_baseline_ms[::-1, ::-1]
T2long_baseline_ms = T2long_baseline_ms[::-1, ::-1]
T2short_w_change_ms = T2short_w_change_ms[::-1, ::-1]
T2long_w_change_ms = T2long_w_change_ms[::-1, ::-1]

# currently no side image map recon here
del t1_image

# min/max values for display
im_max = np.max(im_w_change)
im_min = np.min(im_w_change)

# show initial images and pathology in image space
v.append(
    pv.ThreeAxisViewer([im_baseline, im_w_change, im_w_change - im_baseline],
                       imshow_kwargs=[{
                           'cmap': 'viridis',
                           'vmax': im_max,
                           'vmin': im_min
                       }, {
                           'cmap': 'viridis',
                           'vmax': im_max,
                           'vmin': im_min
                       }, {
                           'cmap': 'viridis',
                       }]))

# --------------------------------------------------------------------------
# simulate data

# move image to GPU
im_baseline = cp.asarray(im_baseline.astype(np.complex128))
im_w_change = cp.asarray(im_w_change.astype(np.complex128))
T2long_baseline_ms = cp.asarray(T2long_baseline_ms)
T2short_baseline_ms = cp.asarray(T2short_baseline_ms)
T2long_w_change_ms = cp.asarray(T2long_w_change_ms)
T2short_w_change_ms = cp.asarray(T2short_w_change_ms)

# init vars for ideal observer analysis
snr = cp.zeros(len(gradient_strengths), dtype=float)
statistic = cp.zeros(len(gradient_strengths), dtype=float)

# for checking and reconstruction
data_im_w_change_for_check = []
data_expect_im_w_change_for_check = []
data_im_baseline_for_check = []
data_expect_im_baseline_for_check = []
k_coords_1_cm_for_check = []

# loop over pulse seq setups/trajectories with different max gradients
for g, grad in enumerate(gradient_strengths):

    # simulation using nufft
    if ft == 'nufft':
        # k-space coordinates from gradient trace files in 1/cm
        k_coords_1_cm = tpi_kspace_coords_1_cm_scanner(grad, data_root_dir,
                                                       const_readout_time)
        k_coords_1_cm_for_check.append(k_coords_1_cm)
        # coords wrt to discrete grid
        k_coords_n = k_coords_1_cm * field_of_view_cm
        print(
            f'k coords shape (num_time_samples, num_readouts, num_space_dims): {k_coords_n.shape}'
        )

        data_im_baseline = []
        data_im_w_change = []

        # split nufft into chunks to avoid GPU RAM overflow
        for i_chunk, k_inds in enumerate(
                # kspace coords are ordered in time
                np.array_split(np.arange(k_coords_n.shape[0]),
                               num_sim_chunks)):

            # baseline model
            data_model = NUFFT_T2Biexp_Model(
                sim_shape, k_coords_n[k_inds, ...], T2short_baseline_ms,
                T2long_baseline_ms, acq_sampling_time_ms, time_bin_width_ms,
                echo_time_ms + k_inds[0] * acq_sampling_time_ms)

            data_operator = data_model.get_operator_w_decay_model()

            data_im_baseline_chunk = data_operator(im_baseline)
            data_im_baseline.append(data_im_baseline_chunk)
            del data_model, data_operator

            # model with change
            data_model = NUFFT_T2Biexp_Model(
                sim_shape, k_coords_n[k_inds, ...], T2short_w_change_ms,
                T2long_w_change_ms, acq_sampling_time_ms, time_bin_width_ms,
                echo_time_ms + k_inds[0] * acq_sampling_time_ms)

            data_operator = data_model.get_operator_w_decay_model()

            data_im_w_change_chunk = data_operator(im_w_change)
            data_im_w_change.append(data_im_w_change_chunk)
            del data_model, data_operator

        # concatenate all data chunks
        data_im_baseline = cp.concatenate(data_im_baseline)
        data_im_w_change = cp.concatenate(data_im_w_change)

    # simulation using approximative fft
    elif ft == 'fft':
        shorter_readout_factor = grad / gradient_strengths[0]

        # baseline
        data_im_baseline, k_mask = tpi_t2biexp_fft(
            im_baseline, T2short_baseline_ms, T2long_baseline_ms,
            grid_shape[0], echo_time_ms, grid_shape[0] // 2,
            shorter_readout_factor)

        # change
        data_im_w_change, k_mask = tpi_t2biexp_fft(
            im_w_change, T2short_w_change_ms, T2long_w_change_ms,
            grid_shape[0], echo_time_ms, grid_shape[0] // 2,
            shorter_readout_factor)

        # account for different effective noise due to different number of samples
        # with higher gradient/readout
        noise_level *= np.sqrt(shorter_readout_factor)

        temp = data_im_w_change[(np.logical_not(k_mask))]
        print(np.any(np.abs(temp)))
        temp = data_im_baseline[(np.logical_not(k_mask))]
        print(np.any(np.abs(temp)))

    # data expectation
    data_expect_im_baseline = data_im_baseline.copy()
    data_expect_im_w_change = data_im_w_change.copy()

    # absolute noise level
    noise_level_abs = noise_level * cp.abs(data_im_baseline.max())
    # add noise to the data
    data_im_baseline += noise_level_abs * (
        cp.random.randn(*data_im_baseline.shape) +
        1j * cp.random.randn(*data_im_baseline.shape))
    data_im_w_change += noise_level_abs * (
        cp.random.randn(*data_im_w_change.shape) +
        1j * cp.random.randn(*data_im_w_change.shape))

    # if fft mask the noise outside of the sphere
    if ft == 'fft':
        data_im_baseline *= k_mask
        data_im_w_change *= k_mask

    # save for visualizing and checking
    data_im_w_change_for_check.append(data_im_w_change)
    data_expect_im_w_change_for_check.append(data_expect_im_w_change)
    data_im_baseline_for_check.append(data_im_baseline)
    data_expect_im_baseline_for_check.append(data_expect_im_baseline)

    # compute ideal observer SNR and statistic
    snr[g] = ideal_observer_snr(data_expect_im_baseline,
                                data_expect_im_w_change, noise_level_abs**2)
    statistic[g] = ideal_observer_statistic(
        data_im_w_change, data_expect_im_baseline, data_expect_im_w_change,
        noise_level_abs**2)

# --------------------------------------------------------------------------
# print ideal observer SNR and statistic

np.set_printoptions(precision=2)  # , floatmode='unique')
print(f"\n Ideal observer SNR for gradients {gradient_strengths}:\n {snr}")
grads = np.array(gradient_strengths)
ind = np.argwhere(grads == 16)[0][0]
print(f"\n Ratio with respect to gradient 16:\n {snr/snr[ind]}")

print(
    f"\n Ideal observer statistic for gradients {gradient_strengths}:\n {statistic}"
)
print(f' statistic relative to gradient 16 {statistic/statistic[ind]}')

# --------------------------------------------------------------------------
# Visualize data for the last gradient, mostly for checking

if ft == 'nufft':
    # k-space magnitude
    k_coords_abs_1_cm = [
        np.linalg.norm(k, axis=-1) for k in k_coords_1_cm_for_check
    ]

    # flat data sorted according to time bins
    for g, grad in enumerate(gradient_strengths):
        data_diff_expect = data_expect_im_baseline_for_check[g] - \
            data_expect_im_w_change_for_check[g]
        data_diff = data_im_baseline_for_check[g] - data_im_w_change_for_check[
            g]
        data_show = cp.asnumpy(cp.abs(data_diff_expect))
        plt.figure(), plt.plot(
            k_coords_abs_1_cm[g].reshape(-1), data_show), plt.suptitle(
                f'Expectation of data difference with nufft, grad {grad}')

        data_show = cp.asnumpy(cp.abs(data_diff))
        plt.figure(), plt.plot(k_coords_abs_1_cm[g].reshape(-1),
                               data_show), plt.suptitle(
                                   f'Data difference with nufft, grad {grad}')

    # show first and last data on the same graph
    data_diff_expect_0 = data_expect_im_baseline_for_check[0] - \
            data_expect_im_w_change_for_check[0]
    data_diff_expect_abs_0 = cp.asnumpy(cp.abs(data_diff_expect_0))
    data_diff_expect_last = data_expect_im_baseline_for_check[-1] - \
            data_expect_im_w_change_for_check[-1]
    data_diff_expect_abs_last = cp.asnumpy(cp.abs(data_diff_expect_last))

    fig, ax = plt.subplots(figsize=(7.5, 6))
    plt.xlabel('|k| [1/cm]')
    plt.ylabel('signal [a.u.]')
    plt.plot(
        k_coords_abs_1_cm[-1].ravel(),
        data_diff_expect_abs_last,
        label=f'{gradient_strengths[-1]*0.01} G/cm')
    plt.plot(
        k_coords_abs_1_cm[0].ravel(),
        data_diff_expect_abs_0,
        label=f'{gradient_strengths[0]*0.01} G/cm')
    plt.legend()
    #ax.set_title(f'{pathology}')

elif ft == 'fft':
    example = 1
    # show only the last gradient, easier to visualize
    data_diff_expect = data_expect_im_baseline_for_check[
        example] - data_expect_im_w_change_for_check[example]
    data_diff = data_im_baseline_for_check[
        example] - data_im_w_change_for_check[example]

    # show noiseless log baseline data, with and without T2* decay
    im1 = cp.asnumpy(
        cp.abs(cp.fft.fftshift(data_expect_im_baseline_for_check[example])))
    im2 = cp.asnumpy(cp.abs(cp.fft.fftshift(cp.fft.fftn(im_baseline))))
    v.append(
        pv.ThreeAxisViewer(
            [np.log(im1, where=im1 > 0.),
             np.log(im2, where=im2 > 0.)],
            imshow_kwargs={
                'cmap': 'viridis',
                'vmax': np.log(im1.max())
            }))

    # show noiseless log data with change, with and without T2* decay
    im1 = cp.asnumpy(
        cp.abs(cp.fft.fftshift(data_expect_im_w_change_for_check[example])))
    im2 = cp.asnumpy(cp.abs(cp.fft.fftshift(cp.fft.fftn(im_w_change))))
    v.append(
        pv.ThreeAxisViewer(
            [np.log(im1, where=im1 > 0.),
             np.log(im2, where=im2 > 0.)],
            imshow_kwargs={
                'cmap': 'viridis',
                'vmax': np.log(im1.max())
            }))

    # show log data diff, expectation and noisy
    im1 = cp.asnumpy(cp.abs(cp.fft.fftshift(data_diff_expect)))
    im2 = cp.asnumpy(cp.abs(cp.fft.fftshift(data_diff)))
    v.append(
        pv.ThreeAxisViewer(
            [np.log(im1, where=im1 > 0.),
             np.log(im2, where=im2 > 0.)],
            imshow_kwargs={'cmap': 'viridis'}))

# --------------------------------------------------------------------
# don't spend time on reconstructing images
if no_recon:
    sys.exit()

# --------------------------------------------------------------------
# Reconstruct and visualize

if ft == 'fft':
    # reconstruct baseline, changed and difference images for the last max gradient using
    # simple ifft with grid_shape matrix size
    op_ifft = sigpy.linop.IFFT(grid_shape, center=False)

    # crop data to nominal resolution
    temp = crop_kspace_data(data_im_baseline_for_check[-1], sim_shape,
                            grid_shape)
    ifft_baseline = op_ifft(temp)

    # crop data to nominal resolution
    temp = crop_kspace_data(data_im_w_change_for_check[-1], sim_shape,
                            grid_shape)
    ifft_w_change = op_ifft(temp)

    # crop data to nominal resolution
    temp = crop_kspace_data(
        data_im_w_change_for_check[-1] - data_im_baseline_for_check[-1],
        sim_shape, grid_shape)
    ifft_diff = op_ifft(temp)

    # show reconstructed images
    im1 = cp.asnumpy(cp.abs(ifft_baseline))
    im2 = cp.asnumpy(cp.abs(ifft_w_change))
    im3 = cp.asnumpy(cp.abs(ifft_diff))
    v.append(
        pv.ThreeAxisViewer([im1, im2, im3],
                           imshow_kwargs={
                               'cmap': 'viridis',
                               'vmax': im_max,
                               'vmin': im_min
                           }))

    # reconstruct pathological images for all the max gradients
    recon_expect_w_change = []
    recon_w_change = []
    for g in np.arange(len(gradient_strengths)):

        # crop data to nominal resolution
        temp = crop_kspace_data(data_expect_im_w_change_for_check[g],
                                sim_shape, grid_shape)
        recon = cp.asnumpy(op_ifft(temp))
        recon_expect_w_change.append(recon)

        # crop data to nominal resolution
        temp = crop_kspace_data(data_im_w_change_for_check[g], sim_shape,
                                grid_shape)
        recon = cp.asnumpy(op_ifft(temp))
        recon_w_change.append(recon)

elif ft == 'nufft':
    # reconstruct baseline, changed and difference images for the last max gradient using
    # simple iterative nufft recon
    recon_expect_w_change = []
    recon_expect_baseline = []
    recon_w_change = []
    max_num_iter = 100

    for g, grad in enumerate(gradient_strengths):
        # normalization factor to account for different matrix sizes between sim and recon,
        # as nufft uses also fft with norm ortho
        norm_factor = 1.
        for d in range(len(grid_shape)):
            norm_factor *= np.sqrt(grid_shape[d] / sim_shape[d])

        # noiseless data with change
        coords = k_coords_1_cm_for_check[g] * field_of_view_cm
        coords = coords.reshape(-1, coords.shape[-1])
        A = sigpy.linop.NUFFT(recon_shape, coords)

        loss = LossL2(recon_shape, data_expect_im_w_change_for_check[g], A)
        im_init = np.concatenate((np.ones(recon_shape).ravel(),
                                  np.zeros(recon_shape).ravel()))
        print(f'reconstructing noiseless data with change for grad {grad}')
        res = sci_opt.minimize(
            loss.loss,
            im_init,
            method='L-BFGS-B',
            jac=loss.loss_grad,
            options={'maxiter': max_num_iter})
        recon = real_to_complex(res.x)
        recon = recon.reshape(recon_shape)
        recon_expect_w_change.append(recon * norm_factor)

        # noisy data with change
        loss = LossL2(recon_shape, data_im_w_change_for_check[g], A)
        im_init = np.concatenate((np.ones(recon_shape).ravel(),
                                  np.zeros(recon_shape).ravel()))
        print(f'reconstructing noisy data with change for grad {grad}')
        res = sci_opt.minimize(
            loss.loss,
            im_init,
            method='L-BFGS-B',
            jac=loss.loss_grad,
            options={'maxiter': max_num_iter})
        recon = real_to_complex(res.x)
        recon = recon.reshape(recon_shape)
        recon_w_change.append(recon * norm_factor)

# show recons from noiseless data with change wrt first grad
ims = [np.abs(x) for x in recon_expect_w_change]
v.append(
    pv.ThreeAxisViewer([*ims],
                       imshow_kwargs={
                           'cmap': 'viridis',
                           'vmax': im_max
                       }))
# show differences of recons from noiseless data with change wrt first grad
show_args = [{
    'cmap': 'viridis',
    'vmax': im_max
}] + [{
    'cmap': 'bwr',
    'vmax': im_max,
    'vmin': -im_max
} for g in range(nb_grads - 1)]

v.append(
    pv.ThreeAxisViewer([ims[0], *(ims[1:] - ims[0])], imshow_kwargs=show_args))
# show differences scaled with adjusted min max
show_args = [{
    'cmap': 'viridis',
    'vmax': im_max
}] + [{
    'cmap': 'bwr',
    'vmax': 0.1 * im_max,
    'vmin': -0.1 * im_max
} for g in range(nb_grads - 1)]
v.append(
    pv.ThreeAxisViewer([ims[0], *(ims[1:] - ims[0])], imshow_kwargs=show_args))

# show noisy patho images for all the gradients
ims = [np.abs(x) for x in recon_w_change]
v.append(
    pv.ThreeAxisViewer([*ims],
                       imshow_kwargs={
                           'cmap': 'viridis',
                           'vmax': im_max
                       }))
# with smoothing
ims = [cp.asnumpy(x) for x in recon_w_change]
ims = [gaussian_filter(x, 0.5) for x in ims]
ims = [np.abs(x) for x in ims]
v.append(
    pv.ThreeAxisViewer([*ims],
                       imshow_kwargs={
                           'cmap': 'viridis',
                           'vmax': im_max
                       }))

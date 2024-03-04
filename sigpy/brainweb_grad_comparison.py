""" Comparison of different TPI gradients:
    - simulation using brainweb with added custom pathology, sigpy NUFFT with added bi-exp T2* decay model
    - regularized NUFFT iterative reconstruction using scipy optimization and sigpy operators,
      with or without decay modeling, gridding recon
    - 1 type of SW-TPI trajectory from gradient trace files, support for some radial-based trajectories, not very general
"""

import argparse
import json
from pathlib import Path
from copy import deepcopy
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndimage
import sigpy
import sys
from pymirc.image_operations import zoom3d

from utils import setup_blob_phantom, setup_brainweb_phantom, kb_rolloff, read_GE_ak_wav, tpi_kspace_coords_1_cm_scanner, radial_goldenmean_kspace_coords_1_cm, radial_density_adapted_kspace_coords_1_cm, radial_random_kspace_coords_1_cm
from utils_sigpy import NUFFTT2starDualEchoModel, projected_gradient_operator, LossL2, real_to_complex

from scipy.ndimage import binary_erosion, zoom
import scipy.optimize as sci_opt
import pymirc.viewer as pv
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--max_num_iter', type=int, default=200)
parser.add_argument('--noise_level', type=float, default=1.4e-2)
parser.add_argument('--no_decay', action='store_true')
parser.add_argument('--show_im', action='store_true')
parser.add_argument('--old', action='store_true')
parser.add_argument('--no_recon', action='store_true')
parser.add_argument('--no_sim', action='store_true')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--beta', type=float, default=1e-2)
parser.add_argument(
    '--patho_change_perc',
    type=float,
    default=0.,
    help='pathological change in percentage of normal tissue')
parser.add_argument(
    '--patho_size_perc',
    type=float,
    default=0.,
    help='pathology max size in percentage of FOV in 1D')
parser.add_argument(
    '--patho_center_perc',
    type=str,
    default='60,50,60',
    help='pathology center in percentages')
parser.add_argument(
    '--pathology',
    type=str,
    default='none',
    help=
    'choices: none, glioma, glioma_treatment, or relevant combinations of: lesion, gm, wm, gwm, cos, multi'
)
parser.add_argument(
    '--traj',
    type=str,
    default='sw-tpi',
    choices=['sw-tpi', 'radial_golden', 'radial_random', 'radial_da'])
parser.add_argument(
    '--recon_type',
    type=str,
    default='simple_iter',
    choices=['simple_iter', 'decay_model_iter'],
    help=
    ' simple_iter is NUFFT with 1-neighbour quadratic regularization, decay_model_iter is the same with the bi-exp T2* decay model'
)

args = parser.parse_args()

max_num_iter = args.max_num_iter
noise_level = args.noise_level
no_decay = args.no_decay
show_im = args.show_im
old = args.old
no_recon = args.no_recon
no_sim = args.no_sim
seed = args.seed
beta = args.beta
patho_change_perc = args.patho_change_perc
patho_size_perc = args.patho_size_perc
patho_center_perc = args.patho_center_perc
pathology = args.pathology
traj = args.traj
recon_type = args.recon_type

#---------------------------------------------------------------
# init
rng = np.random.default_rng(42)
np.random.seed(seed)
patho_center_perc = np.fromstring(patho_center_perc, dtype=np.float32, sep=',')

# not great, special case for simulating a glioma after treatment
if pathology == 'glioma_treatment':
    pathology = 'glioma'
    patho_change_perc *= 0.7

#---------------------------------------------------------------
# fixed parameters

phantom = 'brainweb'

# gradients
if traj == 'sw-tpi':
    gradient_strengths = [16, 32, 48]
elif traj == 'radial_golden' or traj == 'radial_random':
    gradient_strengths = [16, 8]
elif traj == 'radial_da':
    gradient_strengths = [16, 32, 48]

# image shape for data simulation
sim_shape = (256, 256, 256)
num_sim_chunks = 100

# image shape for iterative recons
ishape = (128, 128, 128)
# grid shape for IFFTs
grid_shape = (64, 64, 64)

# downsampling factor from sim to recon
ds_s_i = sim_shape[0] / ishape[0]

field_of_view_cm: float = 22.

# 10us sampling time
acq_sampling_time_ms: float = 0.01

# echo times in ms
echo_time_1_ms = 0.455
echo_time_2_ms = 5.

# signal fraction decaying with short T2* time
short_fraction = 0.6

# corresponds to delta k for a regridded k-space bin
# (central part sampled radially)
time_bin_width_ms: float = 0.25

# time sampling step in micro seconds
dt_us = acq_sampling_time_ms * 1000

# twist point for sw-tpi and radial density adapted
p = 0.4

# gamma by 2pi in MHz/T
gamma_by_2pi_MHz_T: float = 11.262

# kmax for 64 matrix (edge)
kmax64_1_cm = 1 / (2 * field_of_view_cm / 64)

# read the data root directory from the config file
with open('.simulation_config.json', 'r') as f:
    data_root_dir: str = json.load(f)['data_root_dir']

# pathology details
patho_desc = f'_size{patho_size_perc:.1f}_change{int(patho_change_perc)}_center{patho_center_perc[0]:.0}+{patho_center_perc[1]:.0}+{patho_center_perc[2]:.0}' if pathology != 'none' else ''

sim_dir = Path(data_root_dir) / (f'sim_{phantom}_patho_{pathology}{patho_desc}'
                                 + ('_no_decay' if no_decay else '') +
                                 (f'_traj{traj}' if traj != 'sw_tpi' else ''))
sim_dir.mkdir(exist_ok=True, parents=True)

recon_dir = Path(data_root_dir) / (
    f'recon_{phantom}_patho_{pathology}{patho_desc}_noise{noise_level:.1E}' +
    ('_no_decay' if no_decay else '') +
    (f'_traj{traj}' if traj != 'sw_tpi' else ''))
recon_dir.mkdir(exist_ok=True, parents=True)

# init for vizualization
v = []

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# setup the image

if no_decay:
    decay_suffix = '_no_decay'
    T2long_ms_csf: float = 1e7
    T2long_ms_gm: float = 1e7
    T2long_ms_wm: float = 1e7
    T2long_ms_other: float = 1e7
    T2short_ms_csf: float = 1e7
    T2short_ms_gm: float = 1e7
    T2short_ms_wm: float = 1e7
    T2short_ms_other: float = 1e7
else:
    decay_suffix = ''
    T2long_ms_csf: float = 50.
    T2long_ms_gm: float = 20.
    T2long_ms_wm: float = 18.
    T2long_ms_other: float = 18
    T2short_ms_csf: float = 50.
    T2short_ms_gm: float = 3.
    T2short_ms_wm: float = 3.
    T2short_ms_other: float = 3.

field_of_view_cm: float = 22.
phantom_data_path: Path = Path(data_root_dir) / 'brainweb54'

sim_file = sim_dir / 'na_gt.npy'
if not sim_file.exists():

    # setup the brainweb phantom with the given pathology
    x, t1_image, T2short_ms, T2long_ms = setup_brainweb_phantom(
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
        csf_na_concentration=1.5,
        gm_na_concentration=0.6,
        wm_na_concentration=0.4,
        other_na_concentration=0.3,
        add_anatomical_mismatch=False,
        add_T2star_bias=False,
        pathology=pathology,
        pathology_size_perc=patho_size_perc,
        pathology_change_perc=patho_change_perc,
        pathology_center_perc=patho_center_perc)

    # reorient the phantom so it shows correctly with pymirc viewer
    x = x[::-1, ::-1]
    T2short_ms = T2short_ms[::-1, ::-1]
    T2long_ms = T2long_ms[::-1, ::-1]
    del t1_image

    # use the perfect image as prior
    t1_image = x.copy()

    # move image to GPU
    x = cp.asarray(x.astype(np.complex128))

    true_ratio_image_short = cp.array(
        np.exp(-(echo_time_2_ms - echo_time_1_ms) / T2short_ms))
    true_ratio_image_long = cp.array(
        np.exp(-(echo_time_2_ms - echo_time_1_ms) / T2long_ms))

    cp.save(sim_dir / 'na_gt.npy', x)
    cp.save(sim_dir / 't1.npy', t1_image)
    cp.save(sim_dir / 'true_ratio_short.npy', true_ratio_image_short)
    cp.save(sim_dir / 'true_ratio_long.npy', true_ratio_image_long)
else:
    print('loading sim im ')
    with open(sim_file, 'r') as f:
        x = cp.load(sim_file)
        t1_image = cp.load(sim_dir / 't1.npy')
        true_ratio_image_short = cp.load(sim_dir / 'true_ratio_short.npy')
        true_ratio_image_long = cp.load(sim_dir / 'true_ratio_long.npy')

# phantom at nominal resolution
im_nominal_res = zoom3d(cp.asnumpy(x).real, 1. / ds_s_i)
if show_im:
    v.append(
        pv.ThreeAxisViewer(im_nominal_res, imshow_kwargs={'cmap': 'viridis'}))

if no_sim:
    sys.exit()

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# Simulation and reconstruction
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

recon_te1_basic = cp.zeros((len(gradient_strengths), *ishape), x.dtype)
recon_te1_bfgs = cp.zeros((len(gradient_strengths), *ishape), x.dtype)
for g, grad in enumerate(gradient_strengths):
    #--------------------------------------------------------------------------
    # data simulation

    # trajectory
    if traj == 'sw-tpi':
        k_1_cm = tpi_kspace_coords_1_cm_scanner(grad, data_root_dir)
    elif traj == 'radial_golden':
        k_1_cm = radial_goldenmean_kspace_coords_1_cm(grad / 100.)
    elif traj == 'radial_random':
        k_1_cm = radial_random_kspace_coords_1_cm(grad / 100.)
    elif traj == 'radial_da':
        k_1_cm = radial_density_adapted_kspace_coords_1_cm(gradient_strength=grad/100., k_max_1_cm=kmax64_1_cm, dt=dt_us, nb_spokes=3162, p=p)

    k_1_cm_abs = np.linalg.norm(k_1_cm, axis=2)
    #    print(f'readout kmax .: {k_1_cm_abs.max():.2f} 1/cm')
    #    print(f'64 kmax      .: {kmax64_1_cm:.2f} 1/cm')

    # data simulation
    outfile1 = sim_dir / f'data_echo_1_noiseless_grad{grad}.npy'
    outfile2 = sim_dir / f'data_echo_2_noiseless_grad{grad}.npy'
    if not outfile1.exists() or not outfile2.exists():

        data_echo_1 = []
        data_echo_2 = []

        for i_chunk, k_inds in enumerate(
                np.array_split(np.arange(k_1_cm.shape[0]), num_sim_chunks)):
            print('simulating data chunk', i_chunk)

            data_model = NUFFTT2starDualEchoModel(
                sim_shape,
                k_1_cm[k_inds, ...],
                field_of_view_cm=field_of_view_cm,
                acq_sampling_time_ms=acq_sampling_time_ms,
                time_bin_width_ms=time_bin_width_ms,
                echo_time_1_ms=echo_time_1_ms + k_inds[0] *
                acq_sampling_time_ms,  # account of acq. offset time of every chunk
                echo_time_2_ms=echo_time_2_ms +
                k_inds[0] * acq_sampling_time_ms
            )  # account of acq. offset time of every chunk

            data_operator_1_short, data_operator_2_short = data_model.get_operators_w_decay_model(
                true_ratio_image_short)
            data_operator_1_long, data_operator_2_long = data_model.get_operators_w_decay_model(
                true_ratio_image_long)

            #--------------------------------------------------------------------------
            # simulate noise-free data
            data_echo_1.append(short_fraction * data_operator_1_short(x) +
                               (1 - short_fraction) * data_operator_1_long(x))
            data_echo_2.append(short_fraction * data_operator_2_short(x) +
                               (1 - short_fraction) * data_operator_2_long(x))

            del data_operator_1_short
            del data_operator_2_short
            del data_operator_1_long
            del data_operator_2_long
            del data_model

        data_echo_1 = cp.concatenate(data_echo_1)
        data_echo_2 = cp.concatenate(data_echo_2)

        # save noiseless simulated data for the 2 TEs
        cp.save(outfile1, data_echo_1)
        cp.save(outfile2, data_echo_2)

    else:
        print('loading sim data ')
        # load noiseless simulated data for the 2 TEs
        data_echo_1 = cp.load(outfile1)
        data_echo_2 = cp.load(outfile2)

    #--------------------------------------------------------------------------
    # skip reconstruction
    if no_recon:
        continue

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    # scale data such that max of DC component is 2
    # should be the same for all gradients
    if g == 0:
        data_scale = 2.0 / float(cp.abs(data_echo_1).max())

    # scale data
    data_echo_1 *= data_scale
    data_echo_2 *= data_scale

    # add noise to the data
    nl = noise_level * cp.abs(data_echo_1.max())
    data_echo_1 += nl * (cp.random.randn(*data_echo_1.shape) +
                         1j * cp.random.randn(*data_echo_1.shape))
    data_echo_2 += nl * (cp.random.randn(*data_echo_2.shape) +
                         1j * cp.random.randn(*data_echo_2.shape))

    d1 = data_echo_1.reshape(k_1_cm.shape[:-1])
    d2 = data_echo_2.reshape(k_1_cm.shape[:-1])

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    # adjoint nufft image reconstruction

    outfile1 = recon_dir / f'recon_te1_adjoint_grad{grad}_seed{seed}.npz'

    if not outfile1.exists():
        # setup the density compensation weights
        abs_k = np.linalg.norm(k_1_cm, axis=-1)
        if traj == 'sw-tpi' or traj == 'radial_da':
            # for the current sw-tpi config p=0.4, oversampling=1.8,
            # so the twist starts approximately at p * kmax64_1_cm / os
            # TO DO improve this, generalize this to other gradient files or configs
            if traj == 'sw-tpi':
                os = 1.8
            else:
                os = 1.
            k_twist = p * kmax64_1_cm / os
            abs_k_twist_ind = np.argmin(np.abs(abs_k[:, 0] - k_twist))
            abs_k_twist = abs_k[abs_k_twist_ind, 0]
            print(f'abs k twist {abs_k_twist} ind {abs_k_twist_ind}')

            dcf = cp.asarray(np.clip(abs_k**2, None, abs_k_twist**2)).ravel()
        elif traj == 'radial_golden' or traj == 'radial_random':
            dcf = cp.asarray(abs_k**2).ravel()

        ifft1 = sigpy.nufft_adjoint(
            data_echo_1 * dcf,
            cp.array(k_1_cm).reshape(-1, 3) * field_of_view_cm, grid_shape)

        # interpolate to recons (128) grid
        ifft1 = ndimage.zoom(
            ifft1, ishape[0] / grid_shape[0], order=1, prefilter=False)

        recon_te1_basic[g] = ifft1  #/ img_scale

        cp.savez(outfile1, recon=recon_te1_basic[g])
    else:
        saved_res = cp.load(outfile1)
        recon_te1_basic[g] = saved_res['recon']

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    # iterative image reconstruction block

    #------------------------------------------------------
    # NUFFT recon with and without decay model
    acq_model = NUFFTT2starDualEchoModel(
        ishape,
        k_1_cm,
        field_of_view_cm=field_of_view_cm,
        acq_sampling_time_ms=acq_sampling_time_ms,
        time_bin_width_ms=time_bin_width_ms,
        echo_time_1_ms=echo_time_1_ms,
        echo_time_2_ms=echo_time_2_ms)

    # common scaling factors for all gradients,
    # computed based on the reference gradient 16
    nufft_norm_file = recon_dir / 'nufft_norm.txt'

    if not nufft_norm_file.exists() and grad == 16:
        # operators
        nufft_wo_decay_1, nufft_wo_decay_2 = acq_model.get_operators_wo_decay_model(
        )
        del nufft_wo_decay_2
        # get a test single echo nufft operator without T2* decay modeling
        # and estimate its norm
        max_eig_nufft_wo_decay = sigpy.app.MaxEig(
            nufft_wo_decay_1.H * nufft_wo_decay_1,
            dtype=cp.complex128,
            device=data_echo_1.device,
            max_iter=30).run()

        with open(nufft_norm_file, 'w') as f:
            f.write(f'{max_eig_nufft_wo_decay}\n')
    else:
        with open(nufft_norm_file, 'r') as f:
            print('loading max eig')
            max_eig_nufft_wo_decay = float(f.readline())

    scale_file = recon_dir / 'scaling_factors.json'
    if not scale_file.exists() and grad == 16:
        # scale the acquisition model such that norm of the single echo operator is 1
        acq_model_scale = 1 / np.sqrt(max_eig_nufft_wo_decay)

        # calculate an effective scaling factor for the reconstructed images
        img_scale = ((sim_shape[0] / ishape[0])**
                     (3 / 2)) * (data_scale / acq_model_scale)

        # save all scaling factors to a file
        with open(scale_file, 'w') as f:
            json.dump({
                'image_scale': img_scale,
                'data_scale': data_scale,
                'acq_model_scale': acq_model_scale
            }, f)
    else:
        with open(scale_file, 'r') as f:
            print('loading scale')
            temp = json.load(f)
            data_scale_saved: float = temp['data_scale']
            img_scale: float = temp['image_scale']
            acq_model_scale: float = temp['acq_model_scale']
            print(f'data_scale {data_scale}')

    # scale the acquisition model as for the reference gradient 16
    acq_model.scale = acq_model_scale
    #
    #    # nufft operator without decay modeling
    nufft_wo_decay_1, nufft_wo_decay_2 = acq_model.get_operators_wo_decay_model(
    )
    del nufft_wo_decay_2

    # nufft operators with true biexp decay modeling
    recon_operator_1_short, recon_operator_2_short = acq_model.get_operators_w_decay_model(
        true_ratio_image_short[::ds_s_i, ::ds_s_i, ::ds_s_i])
    recon_operator_1_long, recon_operator_2_long = acq_model.get_operators_w_decay_model(
        true_ratio_image_long[::ds_s_i, ::ds_s_i, ::ds_s_i])
    nufft_true_biexp = short_fraction * recon_operator_1_short + (
        1 - short_fraction) * recon_operator_1_long
    del recon_operator_2_short, recon_operator_2_long

    # simple quadratic reg with 1 neighbour
    # set up the operator for regularization
    G = (1 / np.sqrt(12)) * sigpy.linop.FiniteDifference(ishape, axes=None)

    if recon_type == 'simple_iter':
        # regularized nufft recon of TE1 without decay model L-BFGS-B
        A = nufft_wo_decay_1
        outfile1 = recon_dir / f'recon_te1_bfgs_beta{beta:.1E}_it{max_num_iter}_grad{grad}_seed{seed}.npz'
    elif recon_type == 'decay_model_iter':
        # regularized nufft recon of TE1 with true decay model L-BFGS-B
        A = nufft_true_biexp
        outfile1 = recon_dir / f'recon_te1_bfgs_true_biexp_beta{beta:.1E}_it{max_num_iter}_grad{grad}_seed{seed}.npz'
    if not outfile1.exists():
        print(f'BFGS recon {recon_type}')
        loss = LossL2(ishape, data_echo_1, A, G)
        #        loss.check_grad(rng)
        #        loss.check_grad(rng)
        #        loss.check_grad(rng, 1.)

        im_init = np.concatenate((np.ones(ishape).ravel(),
                                  np.zeros(ishape).ravel()))
        res = sci_opt.minimize(
            loss.loss,
            im_init,
            method='L-BFGS-B',
            jac=loss.loss_grad,
            args=(beta),
            options={
                'maxiter': max_num_iter,
                'disp': 1
            })

        recon = real_to_complex(res.x)
        recon = recon.reshape(ishape)

        recon_te1_bfgs[g] = cp.asarray(recon / img_scale)

        cp.savez(outfile1, recon=recon_te1_bfgs[g])
    else:
        saved_res = cp.load(outfile1)
        recon_te1_bfgs[g] = saved_res['recon']

# display results
if show_im:
    x_ds = zoom3d(np.abs(cp.asnumpy(x)), 1. / ds_s_i)
    # basic
    ims = [x_ds] + [np.abs(cp.asnumpy(temp)) for temp in recon_te1_basic]
    ims_arg = [{
        'cmap': 'viridis',
        'vmax': im.max() * 0.9 / (1 + 0.01 * i)
    } for i, im in enumerate(ims)]
    v.append(pv.ThreeAxisViewer(ims, imshow_kwargs=ims_arg))
    # BFGS
    ims = [x_ds] + [np.abs(cp.asnumpy(temp)) for temp in recon_te1_bfgs]
    v.append(
        pv.ThreeAxisViewer(
            ims, imshow_kwargs={
                'cmap': 'viridis',
                'vmax': cp.abs(x).max()
            }))

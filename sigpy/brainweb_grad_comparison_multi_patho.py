""" Final optimized total sodium concentration Na MRI simulation and reconstruction for studying different TPI maximum gradient values

    - simulation using BrainWeb with specific chosen pathologies, gradient trace files, sigpy NUFFT

    - several hard-coded pathologies in the same numerical phantom for reducing the number of reconstructions: glioma + cos lesion, white matter lesions of different sizes

    - regularized iterative reconstruction using scipy L-BFGS-B and sigpy operators,
     with and without decay modeling, with and without the structural higher resolution prior

---
Usage example for several white matter lesions with different sizes (size given as percentage of FOV):

brainweb_grad_comparison_multi_patho.py --pathologies lesion1.6+lesion3+lesion5 --recon_type highres_prior_decay_model_iter --beta 1e-3 --seed 42

"""

import argparse
import json
from pathlib import Path
import numpy as np
import cupy as cp
import sigpy
import sys

from utils import setup_brainweb_phantom, tpi_kspace_coords_1_cm_scanner, real_to_complex
from utils_sigpy import NUFFTT2starDualEchoModel, LossL2, projected_gradient_operator
from pymirc.image_operations import zoom3d

import scipy.optimize as sci_opt
import pymirc.viewer as pv
import matplotlib.pyplot as plt
from functools import partial

#--------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    '--max_num_iter',
    type=int,
    default=200,
    help="number of iterations for reconstruction")
parser.add_argument(
    '--noise_level',
    type=float,
    default=1.4e-2,
    help="noise level relative to the DC component in k-space")
parser.add_argument(
    '--no_decay',
    action='store_true',
    help="ignore T2* decay when simulating raw data")
parser.add_argument(
    '--show_im', action='store_true', help='display images and results')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument(
    '--beta', type=float, default=1e-2, help="spatial regularization weight")
parser.add_argument(
    '--recon_type',
    type=str,
    default='decay_model_iter',
    choices=[
        'simple_iter', 'decay_model_iter', 'highres_prior_decay_model_iter',
        'highres_prior_simple_iter'
    ])
parser.add_argument(
    '--pathologies',
    type=str,
    default='glioma+lesion_gwm_cos',
    choices=[
        'glioma+lesion_gwm_cos', 'none', 'glioma_treatment+lesion_gwm_cos',
        'lesion1.6+lesion3+lesion5', 'lesion3.2+lesion4.6+lesion6.6'
    ]
)  # different pathologies are separated by '+', the number next to the lesion is the lesion size in percentage of FOV (first dim)

args = parser.parse_args()

max_num_iter = args.max_num_iter
noise_level = args.noise_level
no_decay = args.no_decay
show_im = args.show_im
old = args.old
seed = args.seed
beta = args.beta
recon_type = args.recon_type
pathologies = args.pathologies

rng = cp.random.default_rng(seed)

#---------------------------------------------------------------
# fixed parameters
phantom = 'brainweb'

# gradients
gradient_strengths = [16, 32, 48]

# image shape for data simulation
sim_shape = (256, 256, 256)
# do simulation in chunks to decrease GPU RAM usage
num_sim_chunks = 10

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

# gamma by 2pi in MHz/T
gamma_by_2pi_MHz_T: float = 11.262

# kmax for 64 matrix (edge)
kmax64_1_cm = 1 / (2 * field_of_view_cm / 64)

# read the data root directory from the config file
with open('.simulation_config.json', 'r') as f:
    data_root_dir: str = json.load(f)['data_root_dir']

sim_dir = Path(data_root_dir) / (f'final_sim_{phantom}_patho_{pathologies}' +
                                 ('_no_decay' if no_decay else ''))
sim_dir.mkdir(exist_ok=True, parents=True)

recon_dir = Path(data_root_dir) / (
    f'final_recon_{phantom}_patho_{pathologies}_noise{noise_level:.1E}' +
    ('_no_decay' if no_decay else ''))

recon_dir.mkdir(exist_ok=True, parents=True)

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

# helper syntax to avoid repeating the same parameter values in different calls
shorter_setup_brainweb = partial(
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
    csf_na_concentration=1.5,
    gm_na_concentration=0.6,
    wm_na_concentration=0.4,
    other_na_concentration=0.3,
    add_anatomical_mismatch=False,
    add_T2star_bias=False)

sim_file = sim_dir / 'na_gt.npy'
if not sim_file.exists():

    # setup the phantom without pathology
    x_nopatho, t1_image_nopatho, T2short_ms_nopatho, T2long_ms_nopatho = shorter_setup_brainweb(
    )

    if pathologies == 'glioma+lesion_gwm_cos' or pathologies == 'glioma_treatment+lesion_gwm_cos':
        # setup the brainweb phantom with the given pathology
        x_glioma, t1_image, T2short_ms, T2long_ms = shorter_setup_brainweb(
            pathology='glioma', pathology_change_perc=200)

        x_cos, dummy0, dummy1, dummy2 = shorter_setup_brainweb(
            pathology='lesion_gwm_cos',
            pathology_size_perc=20,
            pathology_center_perc=np.array([63.6, 50, 59.1]))

        if 'treatment' in pathologies:
            x_glioma_treatment, t1_image_treatment, T2short_ms_treatment, T2long_ms_treatment = shorter_setup_brainweb(
                pathology='glioma', pathology_change_perc=(200 * 0.7))

            x = x_glioma_treatment + x_cos - x_nopatho
            x_glioma_only = x_glioma_treatment - x_glioma
            x_cos_only = x_cos - x_nopatho
            x_nopatho = x_glioma + x_cos - x_nopatho
            assert (not np.any(x < 0.))

            x_glioma_only = x_glioma_only[::-1, ::-1]
            x_cos_only = x_cos_only[::-1, ::-1]
            # these should not change wrt to glioma before treatment
            del t1_image_treatment, T2short_ms_treatment, T2long_ms_treatment

            x_glioma_only = cp.asarray(x_glioma_only.astype(np.complex128))
            x_cos_only = cp.asarray(x_cos_only.astype(np.complex128))
            cp.save(sim_dir / 'glioma_diff.npy', x_glioma_only)
            cp.save(sim_dir / 'lesion_gwm_cos_diff.npy', x_cos_only)

        else:
            x = x_glioma + x_cos - x_nopatho
            x_glioma_only = x_glioma - x_nopatho
            x_cos_only = x_cos - x_nopatho
            assert (not np.any(x < 0.))

            x_glioma_only = x_glioma_only[::-1, ::-1]
            x_cos_only = x_cos_only[::-1, ::-1]

            x_glioma_only = cp.asarray(x_glioma_only.astype(np.complex128))
            x_cos_only = cp.asarray(x_cos_only.astype(np.complex128))
            cp.save(sim_dir / 'glioma_diff.npy', x_glioma_only)
            cp.save(sim_dir / 'lesion_gwm_cos_diff.npy', x_cos_only)

    elif pathologies == 'none':
        x = x_nopatho
        T2short_ms = T2short_ms_nopatho
        T2long_ms = T2long_ms_nopatho

    elif pathologies == 'lesion1.6+lesion3+lesion5':
        x_les1, t1_image, T2short_ms, T2long_ms = shorter_setup_brainweb(
            pathology='lesion_wm',
            pathology_change_perc=170,
            pathology_size_perc=1.6,
            pathology_center_perc=np.array([39, 65, 53]))

        x_les2, t1_image, T2short_ms, T2long_ms = shorter_setup_brainweb(
            pathology='lesion_wm',
            pathology_change_perc=170,
            pathology_size_perc=3.,
            pathology_center_perc=np.array([41, 56, 60]))

        x_les3, t1_image, T2short_ms, T2long_ms = shorter_setup_brainweb(
            pathology='lesion_wm',
            pathology_change_perc=170,
            pathology_size_perc=5.,
            pathology_center_perc=np.array([36, 48, 56]))

        x = x_nopatho + x_les1 - x_nopatho + x_les2 - x_nopatho + x_les3 - x_nopatho
        assert (np.all(x >= 0.))

    elif pathologies == 'lesion3.2+lesion4.6+lesion6.6':
        x_les1, t1_image, T2short_ms, T2long_ms = shorter_setup_brainweb(
            pathology='lesion_wm',
            pathology_change_perc=21.5,
            pathology_size_perc=3.2,
            pathology_center_perc=np.array([39, 65, 53]))

        x_les2, t1_image, T2short_ms, T2long_ms = shorter_setup_brainweb(
            pathology='lesion_wm',
            pathology_change_perc=46.8,
            pathology_size_perc=4.6,
            pathology_center_perc=np.array([41, 56, 60]))

        x_les3, t1_image, T2short_ms, T2long_ms = shorter_setup_brainweb(
            pathology='lesion_wm',
            pathology_change_perc=74,
            pathology_size_perc=6.6,
            pathology_center_perc=np.array([36, 48, 56]))

        x = x_nopatho + x_les1 - x_nopatho + x_les2 - x_nopatho + x_les3 - x_nopatho
        assert (np.all(x >= 0.))

    # reorient the phantom so it shows correctly with pymirc viewer
    x = x[::-1, ::-1]
    T2short_ms = T2short_ms[::-1, ::-1]
    T2long_ms = T2long_ms[::-1, ::-1]

    # use the perfect image as prior
    t1_image = x.copy()

    # move image to GPU
    x = cp.asarray(x.astype(np.complex128))

    true_ratio_image_short = cp.array(
        np.exp(-(echo_time_2_ms - echo_time_1_ms) / T2short_ms))
    true_ratio_image_long = cp.array(
        np.exp(-(echo_time_2_ms - echo_time_1_ms) / T2long_ms))

    # save to disk
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

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# Simulation and reconstruction
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

recon_te1_bfgs = cp.zeros((len(gradient_strengths), *ishape), x.dtype)

v = []

for g, grad in enumerate(gradient_strengths):
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    # data simulation

    # trajectory
    k_1_cm = tpi_kspace_coords_1_cm_scanner(grad, data_root_dir)
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
    data_echo_1 += nl * (rng.standard_normal(data_echo_1.shape) +
                         1j * rng.standard_normal(data_echo_1.shape))
    data_echo_2 += nl * (rng.standard_normal(data_echo_2.shape) +
                         1j * rng.standard_normal(data_echo_2.shape))

    d1 = data_echo_1.reshape(k_1_cm.shape[:-1])
    d2 = data_echo_2.reshape(k_1_cm.shape[:-1])

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    # iterative image reconstruction block

    #------------------------------------------------------
    # NUFFT recon
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

    # nufft operator without decay modeling
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
        A = nufft_wo_decay_1
        outfile1 = recon_dir / f'recon_te1_bfgs_beta{beta:.1E}_it{max_num_iter}_grad{grad}_seed{seed}.npz'
    elif recon_type == 'decay_model_iter':
        A = nufft_true_biexp
        outfile1 = recon_dir / f'recon_te1_bfgs_true_biexp_beta{beta:.1E}_it{max_num_iter}_grad{grad}_seed{seed}.npz'
    elif recon_type == 'highres_prior_decay_model_iter':
        A = nufft_true_biexp
        t1_image = cp.asnumpy(t1_image)
        t1_image /= np.percentile(t1_image, 99.9)
        prior_image = cp.asarray(zoom3d(t1_image, ishape[0] / sim_shape[0]))
        eta = 0.005
        G = projected_gradient_operator(ishape, prior_image, eta=eta)
        outfile1 = recon_dir / f'recon_te1_bfgs_highres_prior_true_biexp_beta{beta:.1E}_it{max_num_iter}_grad{grad}_seed{seed}.npz'
    elif recon_type == 'highres_prior_simple_iter':
        A = nufft_wo_decay_1
        t1_image = cp.asnumpy(t1_image)
        t1_image /= np.percentile(t1_image, 99.9)
        prior_image = cp.asarray(zoom3d(t1_image, ishape[0] / sim_shape[0]))
        eta = 0.005
        G = projected_gradient_operator(ishape, prior_image, eta=eta)
        outfile1 = recon_dir / f'recon_te1_bfgs_highres_prior_beta{beta:.1E}_it{max_num_iter}_grad{grad}_seed{seed}.npz'

    if not outfile1.exists():
        print(f'BFGS recon {recon_type}')
        loss = LossL2(ishape, data_echo_1, A, G)

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
        print(f"loading recon {recon_type}")
        saved_res = cp.load(outfile1)
        recon_te1_bfgs[g] = saved_res['recon']

#    #--------------------------------------------------------------------------
#    #--------------------------------------------------------------------------
#    # adjoint nufft image reconstruction
#
#    outfile1 = recon_dir / f'recon_te1_adjoint_grad{grad}_seed{seed}.npz'
#
#    if not outfile1.exists():
#        # setup the density compensation weights
#        abs_k = np.linalg.norm(k_1_cm, axis=-1)
#        # twist start should be at 0.4 * kmax_64 / 1.8
#        abs_k_twist_ind = np.argwhere(
#            np.logical_and(abs_k[:, 0] >= 0.3232, abs_k[:, 0] < 0.33))
#        abs_k_twist = abs_k[abs_k_twist_ind[0], 0]
#        print(f'abs k twist {abs_k_twist} ind {abs_k_twist_ind[0]}')
#        print(f'abs k twist {abs_k[114, 0]} ind 114')
#
#        dcf = cp.asarray(np.clip(abs_k**2, None, abs_k_twist**2)).ravel()
#
#        ifft1 = sigpy.nufft_adjoint(
#            data_echo_1 * dcf,
#            cp.array(k_1_cm).reshape(-1, 3) * field_of_view_cm, ishape)
#
#        # interpolate to recons (128) grid
##        ifft1 = ndimage.zoom(
##            ifft1, ishape[0] / grid_shape[0], order=1, prefilter=False)
#
#        #        ifft1_sm = ndimage.gaussian_filter(ifft1, 2.)
#
#        cp.savez(outfile1, recon=ifft1)
#    else:
#        saved_res = cp.load(outfile1)
##        recon_te1_basic[g] = saved_res['recon']

# display results
if show_im:
    ims = [np.abs(cp.asnumpy(x[::ds_s_i, ::ds_s_i, ::ds_s_i]))
           ] + [np.abs(cp.asnumpy(temp)) for temp in recon_te1_bfgs]
    v.append(
        pv.ThreeAxisViewer(
            ims, imshow_kwargs={
                'cmap': 'viridis',
                'vmax': cp.abs(x).max()
            }))

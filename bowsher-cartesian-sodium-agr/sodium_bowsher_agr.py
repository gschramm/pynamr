import numpy as np
import nibabel as nib
import h5py
from pathlib import Path

import matplotlib.pyplot as plt

from scipy.optimize import fmin_l_bfgs_b
from scipy.ndimage import zoom, gaussian_filter

from nearest_neighbors import next_neighbors, nearest_neighbors, is_nearest_neighbor_of
from readout_time import readout_time
from cost_functions import multi_echo_bowsher_cost, multi_echo_bowsher_grad, multi_echo_bowsher_cost_gamma
from cost_functions import multi_echo_bowsher_grad_gamma, multi_echo_bowsher_cost_total
from utils import load_config, load_recon_config

from argparse import ArgumentParser
from datetime import datetime

import pymirc.viewer as pv

parser = ArgumentParser()
parser.add_argument('--cfg', help='config json file', default='config.json')
parser.add_argument('--recon_cfg', help='config json file', default='recon_params.json')
args = parser.parse_args()

cfg = load_config(args.cfg)
recon_cfg = load_recon_config(args.recon_cfg)

niter = recon_cfg.niter
n_outer = recon_cfg.n_outer
bet_recon = recon_cfg.bet_recon
bet_gam = recon_cfg.bet_gam
nnearest = recon_cfg.nnearest
nneigh = recon_cfg.nneigh
method = recon_cfg.method
mr_name = recon_cfg.mr_name
anat_prior = not recon_cfg.no_anat_prior

delta_t = 5.
asym = 0
n = cfg.recon_shape[0]

# %%
# load the data
# -------------

pdir = cfg.recon_dir

if anat_prior:
    odir = Path(pdir) / f'{datetime.now().strftime("%y%m%d-%H%M%S")}__br_{bet_recon:.1E}__bg_{bet_gam:.1E}__nn_{nneigh}__ne_{nnearest}'
else:
    odir = Path(pdir) / f'{datetime.now().strftime("%y%m%d-%H%M%S")}__br_{bet_recon:.1E}__bg_{bet_gam:.1E}__no_anat'

if not odir.exists():
    odir.mkdir(exist_ok=True)

# copy config files
(odir / 'config.json').write_bytes(Path(args.cfg).read_bytes())
(odir / 'recon_params.json').write_bytes(Path(args.recon_cfg).read_bytes())

nechos = 2

t1_vol = np.load(Path(pdir) / mr_name)
sens = np.load(Path(pdir) / f'sens_{n}.npy').astype(
    np.complex128).view('(2,)float')
echo1 = np.load(Path(pdir) / f'echo1_{n}.npy').astype(
    np.complex128).view('(2,)float')
echo2 = np.load(Path(pdir) / f'echo2_{n}.npy').astype(
    np.complex128).view('(2,)float')

ncoils = echo1.shape[0]

signal = np.zeros((ncoils, nechos) + echo1.shape[1:])

for i in range(ncoils):
    signal[i, 0, ...] = echo1[i, ...]
    signal[i, 1, ...] = echo2[i, ...]

# %%
# calc readout times
# ------------------

# setup the frequency array as used in numpy fft
k0, k1, k2 = np.meshgrid(
    np.arange(n) - n // 2 + 0.5,
    np.arange(n) - n // 2 + 0.5,
    np.arange(n) - n // 2 + 0.5)
abs_k = np.sqrt(k0**2 + k1**2 + k2**2)
abs_k = np.fft.fftshift(abs_k)

# rescale abs_k such that k = 1.5 is at r = 32 (the edge)
k_edge = 1.5
abs_k *= k_edge / 32

# calculate the readout times and the k-spaces locations that
# are read at a given time
t_read_3 = 1000 * readout_time(abs_k)

n_readout_bins = 32

k_1d = np.linspace(0, k_edge, n_readout_bins + 1)

readout_inds = []
tr = np.zeros(n_readout_bins)
t_read_3_binned = np.zeros(t_read_3.shape)

read_out_img = np.zeros((n, n, n))

for i in range(n_readout_bins):
    k_start = k_1d[i]
    k_end = k_1d[i + 1]
    rinds = np.where(np.logical_and(abs_k >= k_start, abs_k <= k_end))

    tr[i] = t_read_3[rinds].mean()
    t_read_3_binned[rinds] = tr[i]
    readout_inds.append(rinds)
    read_out_img[rinds] = i + 1

#------------
#------------

kmask = np.zeros(signal.shape)
for j in range(ncoils):
    for i in range(nechos):
        kmask[j, i, ..., 0] = (read_out_img > 0).astype(np.float64)
        kmask[j, i, ..., 1] = (read_out_img > 0).astype(np.float64)

# multiply signal with readout mask
signal *= kmask
abs_signal = np.linalg.norm(signal, axis=-1)

ifft = np.zeros(signal.shape)
ifft_filtered = np.zeros(signal.shape)
abs_ifft = np.zeros(signal.shape[:-1])
abs_ifft_filtered = np.zeros(signal.shape[:-1])

for j in range(ncoils):
    for i in range(nechos):
        s = signal[j, i, ...].view(dtype=np.complex128).squeeze().copy()
        ifft[j, i, ...] = np.ascontiguousarray(
            np.fft.ifftn(s, norm='ortho').view('(2,)float'))
        ifft_filtered[j, i, ...] = np.ascontiguousarray(
            gaussian_filter(np.fft.ifftn(s, norm='ortho'), 1.4).view('(2,)float'))
        abs_ifft[j, i, ...] = np.linalg.norm(ifft[j, i, ...], axis=-1)
        abs_ifft_filtered[j, i, ...] = np.linalg.norm(ifft_filtered[j, i, ...],
                                                      axis=-1)
# %%
# set up the anatomical prior
# ---------------------------

aimg = t1_vol.copy()

if nneigh == 18:
    s = np.array([[[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                  [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
                  [[0, 1, 0], [1, 1, 1], [0, 1, 0]]])
elif nneigh == 80:
    s = np.array([[[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]],
                  [[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]],
                  [[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1],
                   [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]],
                  [[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]],
                  [[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]])

if anat_prior:
    ninds = np.zeros((np.prod(aimg.shape), nnearest), dtype=np.uint32)
    nearest_neighbors(aimg, s, nnearest, ninds)
else:
    ninds = np.zeros((np.prod(aimg.shape), aimg.ndim), dtype=np.uint32)
    next_neighbors(aimg.shape, ninds)

ninds2 = is_nearest_neighbor_of(ninds)

# %%
# initialize variables
# --------------------

tmp1 = abs_ifft_filtered[:, 0, ...].sum(0)
tmp2 = abs_ifft_filtered[:, 1, ...].sum(0)

Gam_recon = tmp2 / tmp1
Gam_recon[tmp1 < 0.05 * tmp1.max()] = 1
Gam_recon[Gam_recon > 1] = 1

recon = np.zeros(signal.shape[2:])
recon[..., 0] = tmp1

recon_shape = recon.shape
Gam_bounds = ((Gam_recon.shape[0])**3) * [(0.001, 1)]

# rescale signal and initial recon to get an image with max approx 1
scale_fac = 1. / tmp1.max()

signal *= scale_fac
tmp1 *= scale_fac
tmp2 *= scale_fac
recon *= scale_fac

abs_recon = np.linalg.norm(recon, axis=-1)

cost = []

fig1, ax1 = plt.subplots(2, n_outer + 1, figsize=((n_outer + 1) * 3, 6))
vmax = 1.2
ax1[0, 0].imshow(Gam_recon[..., 64], vmin=0, vmax=1, cmap=plt.cm.Greys_r)
ax1[1, 0].imshow(abs_recon[..., 64], vmin=0, vmax=vmax, cmap=plt.cm.Greys_r)

# %% run iterative Bowsher AGR
# ----------------------------

for i in range(n_outer):

    print('LBFGS to optimize for recon')

    recon = recon.flatten()

    cb = lambda x: cost.append(
        multi_echo_bowsher_cost_total(
            x, recon_shape, signal, readout_inds, Gam_recon, tr, delta_t,
            nechos, kmask, bet_recon, bet_gam, ninds, method, sens, asym))

    res = fmin_l_bfgs_b(multi_echo_bowsher_cost,
                        recon,
                        fprime=multi_echo_bowsher_grad,
                        args=(recon_shape, signal, readout_inds, Gam_recon, tr,
                              delta_t, nechos, kmask, bet_recon, ninds, ninds2,
                              method, sens, asym),
                        callback=cb,
                        maxiter=niter,
                        disp=1)

    recon = res[0].reshape(recon_shape)
    abs_recon = np.linalg.norm(recon, axis=-1)

    ax1[1, i + 1].imshow(abs_recon[..., 64],
                         vmin=0,
                         vmax=vmax,
                         cmap=plt.cm.Greys_r)

    #---------------------------------------

    print('LBFGS to optimize for gamma')

    Gam_recon = Gam_recon.flatten()

    cb = lambda x: cost.append(
        multi_echo_bowsher_cost_total(recon, recon_shape, signal, readout_inds,
                                      x, tr, delta_t, nechos, kmask, bet_recon,
                                      bet_gam, ninds, method, sens, asym))

    res = fmin_l_bfgs_b(multi_echo_bowsher_cost_gamma,
                        Gam_recon,
                        fprime=multi_echo_bowsher_grad_gamma,
                        args=(recon_shape, signal, readout_inds, recon, tr,
                              delta_t, nechos, kmask, bet_gam, ninds, ninds2,
                              method, sens, asym),
                        callback=cb,
                        maxiter=niter,
                        bounds=Gam_bounds,
                        disp=1)

    Gam_recon = res[0].reshape(recon_shape[:-1])

    # reset values in low signal regions
    Gam_recon[tmp1 < 0.05 * tmp1.max()] = 1

    ax1[0, i + 1].imshow(Gam_recon[..., 64],
                         vmin=0,
                         vmax=1,
                         cmap=plt.cm.Greys_r)

    fig1.tight_layout()
    fig1.show()
    fig1.savefig(Path(odir) / 'convergence.png')

# %% 
# save the recons to HDF5
# -----------------------

output_file = Path(odir) / 'recons.h5'
with h5py.File(output_file, 'w') as hf:
    grp = hf.create_group('images')
    grp.create_dataset('Gam_recon', data=Gam_recon)
    grp.create_dataset('recon', data=recon)
    grp.create_dataset('abs_recon', data=abs_recon)
    grp.create_dataset('prior_image', data=aimg)
    grp.create_dataset('cost', data=cost)

# %% 
# save the recons to NIFTI
# ------------------------

output_affine_ras = np.diag([cfg.sodium_fov_mm/n,cfg.sodium_fov_mm/n,cfg.sodium_fov_mm/n,1])
output_affine_ras[:-1,-1] = 0.5*(-220 + cfg.sodium_fov_mm/n)

nib.save(nib.Nifti1Image(abs_recon, output_affine_ras), odir  / f'AGR_{odir.name}.nii')
nib.save(nib.Nifti1Image(Gam_recon, output_affine_ras), odir  / f'GAM_{odir.name}.nii')
nib.save(nib.Nifti1Image(aimg, output_affine_ras), odir  / f'T1_proton_aligned.nii')

# save the interpolated SOS images
sos1 = np.abs(np.flip(np.fromfile(cfg.TE05soskw0_filename, dtype = np.complex64).reshape(cfg.data_shape).swapaxes(0,2), (1,2)))
sos2 = np.abs(np.flip(np.fromfile(cfg.TE05soskw1_filename, dtype = np.complex64).reshape(cfg.data_shape).swapaxes(0,2), (1,2)))

sos1 = zoom(sos1, np.array(cfg.recon_shape)/ np.array(cfg.data_shape), order = 1, prefilter = False)
sos2 = zoom(sos2, np.array(cfg.recon_shape)/ np.array(cfg.data_shape), order = 1, prefilter = False)

nib.save(nib.Nifti1Image(sos1, output_affine_ras),  odir  / f'sos_virt_channels_no_smoothing.nii')
nib.save(nib.Nifti1Image(sos2, output_affine_ras),  odir  / f'sos_virt_channels_smoothed.nii')


# %%
# show the results
# ----------------

# scale sos images
scale_fac = abs_recon.sum() / sos2.sum()
sos1 *= scale_fac
sos2 *= scale_fac

vmax = np.percentile(abs_recon, 99.9)

ims2 = [{
    'cmap': plt.cm.Greys_r,
    'vmax': np.percentile(aimg, 99.9)
}] + 3 * [{
    'cmap': plt.cm.Greys_r,
    'vmin': 0,
    'vmax': vmax
}] + [{
    'cmap': plt.cm.Greys_r,
    'vmin': 0,
    'vmax': 1.
}]
vi2 = pv.ThreeAxisViewer([
    np.flip(x, (0, 1))
    for x in [aimg, sos1, sos2, abs_recon, Gam_recon]
],
                         imshow_kwargs=ims2)
vi2.fig.savefig(Path(odir) / 'recons.png')

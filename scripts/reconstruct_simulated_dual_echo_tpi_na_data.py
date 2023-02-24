import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.optimize import fmin_l_bfgs_b

from utils import TriliniearKSpaceRegridder, tpi_sampling_density
from pymirc.image_operations import zoom3d

import pymirc.viewer as pv

# hacks to include old cost functions
import sys
if '..' not in sys.path:
    sys.path.append('..')

from cost_functions import multi_echo_bowsher_cost, multi_echo_bowsher_grad, multi_echo_bowsher_cost_gamma
from cost_functions import multi_echo_bowsher_grad_gamma, multi_echo_bowsher_cost_total
from nearest_neighbors import nearest_neighbors, is_nearest_neighbor_of

#-------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--gradient_strength',
                    type=int,
                    default=16,
                    choices=[16, 24, 32, 48])
parser.add_argument('--noise_level', type=float, default=0)
parser.add_argument('--gridded_data_matrix_size',
                    type=int,
                    default=128,
                    choices=[64, 128, 256])
parser.add_argument('--show_trajectory', action='store_true')
parser.add_argument('--beta_recon', type=float, default=1.)
parser.add_argument('--beta_gamma', type=float, default=10.)
parser.add_argument('--num_outer', type=int, default=10)
parser.add_argument('--num_inner', type=int, default=20)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--phantom',
                    type=str,
                    default='brainweb',
                    choices=['brainweb', 'blob'])
parser.add_argument('--no_decay', action='store_true')
args = parser.parse_args()

# input parameters
gridded_data_matrix_size: int = args.gridded_data_matrix_size
noise_level: float = args.noise_level
gradient_strength: int = args.gradient_strength
show_trajectory: bool = args.show_trajectory
beta_recon: float = args.beta_recon
beta_gamma: float = args.beta_gamma
num_outer: int = args.num_outer
num_inner: int = args.num_inner
seed: int = args.seed
phantom: str = args.phantom

if args.no_decay:
    decay_suffix = '_no_decay'
else:
    decay_suffix = ''

#-------------------------------------------------------------------------
# fixed parameters
num_echos: int = 2
delta_t: float = 5.
num_nearest: int = 4
method: int = 0
asym: int = 0
readout_bin_width_ms: float = 0.5
interpolation: str = 'trilinear'

#-------------------------------------------------------------------------
np.random.seed(seed)

# create the output directory and save the input parameters
i_out = 0
output_dir = Path(
    f'run/g_{gradient_strength}_br_{beta_recon:.2e}_bg_{beta_gamma:.2e}_n_{noise_level:.2e}_s_{seed}_{i_out:03}'
)

while output_dir.exists():
    i_out += 1
    output_dir = Path(
        f'run/g_{gradient_strength}_br_{beta_recon:.2e}_bg_{beta_gamma:.2e}_n_{noise_level:.2e}_s_{seed}_{i_out:03}'
    )

output_dir.mkdir(exist_ok=True, parents=True)

# save input configuration
with open(output_dir / 'config.json', 'w') as f:
    json.dump(vars(args), f)

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# restore the saved noise-free nufft data from file

if gradient_strength == 16:
    gradient_file: str = '/data/sodium_mr/tpi_gradients/n28p4dt10g16_23Na_v1'
elif gradient_strength == 24:
    gradient_file: str = '/data/sodium_mr/tpi_gradients/n28p4dt10g24f23'
elif gradient_strength == 32:
    gradient_file: str = '/data/sodium_mr/tpi_gradients/n28p4dt10g32f23'
elif gradient_strength == 48:
    gradient_file: str = '/data/sodium_mr/tpi_gradients/n28p4dt10g48f23'
else:
    raise ValueError

simulated_data_path = Path(
    '/data'
) / 'sodium_mr' / f'{phantom}_{Path(gradient_file).name}{decay_suffix}'

print(simulated_data_path.name)

data = np.load(simulated_data_path / 'simulated_nufft_data.npz')

nonuniform_data_long_echo_1 = data['nonuniform_data_long_echo_1']
nonuniform_data_long_echo_2 = data['nonuniform_data_long_echo_2']
nonuniform_data_short_echo_1 = data['nonuniform_data_short_echo_1']
nonuniform_data_short_echo_2 = data['nonuniform_data_short_echo_2']
k0 = data['k0']
k1 = data['k1']
k2 = data['k2']
kp = float(data['kp'])
kmax = float(data['kmax'])
kx = data['kx']
ky = data['ky']
kz = data['kz']
t_echo_1_ms = data['t_echo_1_ms']
t_echo_2_ms = data['t_echo_2_ms']
field_of_view_cm = data['field_of_view_cm']
na_image = data['na_image']
t1_image = data['t1_image']
T2short_ms = data['T2short_ms']
T2long_ms = data['T2long_ms']

simulation_matrix_size = na_image.shape[0]

abs_k = np.sqrt(kx**2 + ky**2 + kz**2)
abs_k_spoke = abs_k[-1, :]
t_spoke_ms = np.arange(abs_k.shape[1]) / 100

# setup interpolation functions for abs_k(t) and t(abs_k)
k_of_t = interp1d(t_spoke_ms, abs_k_spoke)
t_of_k = interp1d(abs_k_spoke, t_spoke_ms)

if show_trajectory:
    fig = plt.figure(figsize=(8, 8))
    i = 1500
    jmax = np.where(abs_k[i, :] <= kp)[0].max()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter3D(kx[i, :jmax], ky[i, :jmax], kz[i, :jmax], s=0.5)
    ax.scatter3D(kx[i, jmax:], ky[i, jmax:], kz[i, jmax:], s=0.5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    fig.tight_layout()
    fig.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(abs_k[-1, :])
    ax2.axhline(kmax)
    ax2.axhline(kp)
    fig2.show()

#-------------------------------------------------------------------------
# add noise to the non-uniform fft data
data_size = nonuniform_data_long_echo_1.size
noisy_nonuniform_data_long_echo_1 = nonuniform_data_long_echo_1 + noise_level * (
    np.random.randn(data_size) + 1j * np.random.randn(data_size))
noisy_nonuniform_data_long_echo_2 = nonuniform_data_long_echo_2 + noise_level * (
    np.random.randn(data_size) + 1j * np.random.randn(data_size))

noisy_nonuniform_data_short_echo_1 = nonuniform_data_short_echo_1 + noise_level * (
    np.random.randn(data_size) + 1j * np.random.randn(data_size))
noisy_nonuniform_data_short_echo_2 = nonuniform_data_short_echo_2 + noise_level * (
    np.random.randn(data_size) + 1j * np.random.randn(data_size))

#-------------------------------------------------------------------------

delta_k = 1 / field_of_view_cm

# calculate vector of TPI sampling densities for all kspace points in the non-uniform data
sampling_density = tpi_sampling_density(k0.ravel(), k1.ravel(), k2.ravel(), kp)

regridder = TriliniearKSpaceRegridder(gridded_data_matrix_size, delta_k,
                                      k0.ravel(), k1.ravel(), k2.ravel(),
                                      sampling_density, kmax)

regridded_data_echo_1 = regridder.regrid(
    0.6 * noisy_nonuniform_data_short_echo_1 +
    0.4 * noisy_nonuniform_data_long_echo_1)

regridded_data_echo_2 = regridder.regrid(
    0.6 * noisy_nonuniform_data_short_echo_2 +
    0.4 * noisy_nonuniform_data_long_echo_2)

# correct for the global norm of the FFT operator
nufft_norm = 11626.0
regridded_data_echo_1 /= nufft_norm
regridded_data_echo_2 /= nufft_norm

ifft_echo_1 = np.fft.ifftn(regridded_data_echo_1, norm='ortho')
ifft_echo_2 = np.fft.ifftn(regridded_data_echo_2, norm='ortho')

# interpolate magnitude images to simulation grid size (which can be different from gridded data size)
a = zoom3d(np.abs(ifft_echo_1),
           simulation_matrix_size / gridded_data_matrix_size)
b = zoom3d(np.abs(ifft_echo_2),
           simulation_matrix_size / gridded_data_matrix_size)

#ims = 3 * [{'vmin': 0, 'vmax': 1.2 * na_image.max()}]
#vi = pv.ThreeAxisViewer([na_image, a, b], imshow_kwargs=ims)

#----------------------------------------------------------------------------------------
#--- AGR recon --------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

# create an array for the dual echo (multi_channel data)

signal = np.array([regridded_data_echo_1, regridded_data_echo_2])
# we have to create the "coil" axis for the single coil
signal = np.expand_dims(signal, 0)
# convert the complex signal into a real pseudo complex array
signal = np.ascontiguousarray(signal.view('(2,)float'))

# setup the readout inds for the apodized fft
k_fft = np.fft.fftfreq(gridded_data_matrix_size,
                       d=field_of_view_cm / gridded_data_matrix_size)
K0_fft, K1_fft, K2_fft = np.meshgrid(k_fft, k_fft, k_fft, indexing='ij')
K_abs_fft = np.sqrt(K0_fft**2 + K1_fft**2 + K2_fft**2)

readout_time_image_ms = np.full(K_abs_fft.shape, -1., dtype=np.float64)

mask = (K_abs_fft < abs_k_spoke.max()).astype(float)

# setup a pseudo-complex kspace mask that indicates which elements of kspace where sampled
kmask = np.zeros(signal.shape)
for j in range(1):
    for i in range(2):
        kmask[j, i, ..., 0] = mask
        kmask[j, i, ..., 1] = mask

inds = np.where(mask > 0)
readout_time_image_ms[inds] = t_of_k(K_abs_fft[inds])

readout_bin_image = np.floor(readout_time_image_ms /
                             readout_bin_width_ms).astype(int)

tr = (np.arange(readout_bin_image.max() + 1) + 0.5) * readout_bin_width_ms
readout_inds = []
for i, _ in enumerate(tr):
    readout_inds.append(np.where(readout_bin_image == i))

#vi2 = pv.ThreeAxisViewer(np.fft.fftshift(readout_bin_image))

#-------------------------------------------------------------------------------
# setup an initial guess for Gamma (the ratio between the second and first echo)

# create the han window that we need to multiply to the mask
h_win = interp1d(np.arange(32),
                 np.hanning(64)[32:],
                 fill_value=0,
                 bounds_error=False)
# abs_k was scaled to have the k edge at 32, we have to revert that for the han window
hmask = h_win(K_abs_fft.ravel() * 32 / kmax).reshape(K_abs_fft.shape)

ifft_echo_1_filtered = np.fft.ifftn(hmask * regridded_data_echo_1,
                                    norm='ortho')
ifft_echo_2_filtered = np.fft.ifftn(hmask * regridded_data_echo_2,
                                    norm='ortho')

Gam_bounds = (gridded_data_matrix_size**3) * [(0.001, 1)]
Gam_recon = np.clip(
    np.abs(ifft_echo_2_filtered) / (np.abs(ifft_echo_1_filtered) + 0.001),
    0.001, 1)
Gam_recon[np.abs(ifft_echo_2_filtered) < 0.1 *
          np.abs(ifft_echo_2_filtered).max()] = 1
Gam_recon = gaussian_filter(Gam_recon, 1)

#vi3 = pv.ThreeAxisViewer(Gam_recon)

# setup the anatomical prior image and the nearest neighbors arrays for
# the Bowsher prior
aimg = zoom3d(t1_image, gridded_data_matrix_size / t1_image.shape[0])
# structural element for the 80 nearest neighbots
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

ninds = np.zeros((np.prod(aimg.shape), num_nearest), dtype=np.uint32)
nearest_neighbors(aimg, s, num_nearest, ninds)
ninds2 = is_nearest_neighbor_of(ninds)

# (2) the actual AGR recon
cost = []
# the shape of the pseudo-complex reconstruction
recon_shape = signal.shape[2:]

# squeeze simulated data into a format that we can digest with the AGR recon
# setup initial recon as ifft of first echo
# we need to convert to pseudo-complex array for fmin_l_bfgs
recon = np.ascontiguousarray(ifft_echo_1.view('(2,)float'))

# create a pseudo complex sensitivity array full of real ones
sens = np.zeros((1, ) + recon_shape)
sens[0, ..., 0] = 1

abs_recons = np.zeros((num_outer, ) + ifft_echo_1.shape)
Gam_recons = np.zeros((num_outer, ) + ifft_echo_1.shape)

for i in range(num_outer):
    print('LBFGS to optimize for recon')
    recon = recon.flatten()

    cb = lambda x: cost.append(
        multi_echo_bowsher_cost_total(
            x, recon_shape, signal, readout_inds, Gam_recon, tr, delta_t,
            num_echos, kmask, beta_recon, beta_gamma, ninds, method, sens, asym
        ))

    res = fmin_l_bfgs_b(multi_echo_bowsher_cost,
                        recon,
                        fprime=multi_echo_bowsher_grad,
                        args=(recon_shape, signal, readout_inds, Gam_recon, tr,
                              delta_t, num_echos, kmask, beta_recon, ninds,
                              ninds2, method, sens, asym),
                        callback=cb,
                        maxiter=num_inner,
                        disp=1)

    recon = res[0].reshape(recon_shape)
    abs_recons[i, ...] = np.linalg.norm(recon, axis=-1)

    #---------------------------------------
    print('LBFGS to optimize for gamma')
    Gam_recon = Gam_recon.flatten()

    cb = lambda x: cost.append(
        multi_echo_bowsher_cost_total(
            recon, recon_shape, signal, readout_inds, x, tr, delta_t,
            num_echos, kmask, beta_recon, beta_gamma, ninds, method, sens, asym
        ))

    res = fmin_l_bfgs_b(multi_echo_bowsher_cost_gamma,
                        Gam_recon,
                        fprime=multi_echo_bowsher_grad_gamma,
                        args=(recon_shape, signal, readout_inds, recon, tr,
                              delta_t, num_echos, kmask, beta_gamma, ninds,
                              ninds2, method, sens, asym),
                        callback=cb,
                        maxiter=num_inner,
                        bounds=Gam_bounds,
                        disp=1)

    Gam_recon = res[0].reshape(recon_shape[:-1])
    Gam_recons[i, ...] = Gam_recon

# save the results
np.save(output_dir / 'ifft_echo_1_corr.npy', ifft_echo_1)
np.save(output_dir / 'ifft_echo_2_corr.npy', ifft_echo_2)
np.save(output_dir / 'ifft_echo_1_filtered_corr.npy', ifft_echo_1_filtered)
np.save(output_dir / 'ifft_echo_2_filtered_corr.npy', ifft_echo_2_filtered)
np.save(output_dir / 'agr_na.npy',
        np.squeeze(recon.view(dtype=np.complex128), axis=-1))
np.save(output_dir / 'gamma.npy', Gam_recon)
np.save(output_dir / 'anatomical_prior_image.npy', aimg)
np.save(output_dir / 'true_na_image.npy', na_image)
np.save(output_dir / 't1_image.npy', t1_image)

ims = 3 * [{
    'cmap': plt.cm.Greys_r,
    'vmin': 0,
    'vmax': 2.5
}] + [{
    'cmap': plt.cm.Greys_r,
    'vmin': 0.5,
    'vmax': 1.
}]

vi4 = pv.ThreeAxisViewer([
    np.flip(abs_recons[-1, ...], (0, 1)),
    np.flip(np.abs(ifft_echo_1), (0, 1)),
    np.flip(np.abs(ifft_echo_1_filtered), (0, 1)),
    np.flip(Gam_recons[-1, ...], (0, 1))
],
                         imshow_kwargs=ims)
vi4.fig.savefig(output_dir / '00_screenshot.png')

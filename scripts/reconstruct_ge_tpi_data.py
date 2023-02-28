import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.optimize import fmin_l_bfgs_b

from simulate_dual_echo_tpi_na_data import read_tpi_gradient_files

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
parser.add_argument(
    '--data_dir',
    type=str,
    default='/data/sodium_mr/20230225_MR3_GS_TPI/preprocessed_regridded_data')
parser.add_argument('--beta_recon', type=float, default=3.)
parser.add_argument('--beta_gamma', type=float, default=10.)
parser.add_argument('--num_outer', type=int, default=10)
parser.add_argument('--num_inner', type=int, default=20)
args = parser.parse_args()

# input parameters
beta_recon: float = args.beta_recon
beta_gamma: float = args.beta_gamma
num_outer: int = args.num_outer
num_inner: int = args.num_inner
data_dir: Path = Path(args.data_dir)

show_trajectory: bool = True

na_echo_1_file: Path = data_dir / 'echo_1.npy'
na_echo_2_file: Path = data_dir / 'echo_2.npy'
t1_nifti_file: Path = data_dir / 't1_aligned.npy'

#-------------------------------------------------------------------------
# fixed parameters
num_echos: int = 2
delta_t: float = 5.
num_nearest: int = 4
method: int = 0
asym: int = 0
readout_bin_width_ms: float = 0.5

gradient_strength: int = 16
field_of_view_cm: float = 22.

i_out = 0
output_dir = data_dir.parent / f'recons/br_{beta_recon:.2e}_bg_{beta_gamma:.2e}_{i_out:03}'

while output_dir.exists():
    i_out += 1
    output_dir = data_dir.parent / f'recons/br_{beta_recon:.2e}_bg_{beta_gamma:.2e}_{i_out:03}'

output_dir.mkdir(exist_ok=True, parents=True)

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# restore the saved noise-free nufft data from file

if gradient_strength == 16:
    gradient_file: str = '/data/sodium_mr/tpi_gradients/n28p4dt10g16_23Na_v1'
else:
    raise ValueError

kx, ky, kz, header, n_readouts_per_cone = read_tpi_gradient_files(
    gradient_file)

abs_k = np.sqrt(kx**2 + ky**2 + kz**2)
abs_k_spoke = abs_k[-1, :]
t_spoke_ms = np.arange(abs_k.shape[1]) / 100

# setup interpolation functions for abs_k(t) and t(abs_k)
k_of_t = interp1d(t_spoke_ms, abs_k_spoke)
t_of_k = interp1d(abs_k_spoke, t_spoke_ms)

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

regridded_data_echo_1_phase_corrected = np.load(na_echo_1_file)
regridded_data_echo_2_phase_corrected = np.load(na_echo_2_file)

# IFFT of the regridded data
ifft_echo_1 = np.fft.ifftn(regridded_data_echo_1_phase_corrected, norm='ortho')
ifft_echo_2 = np.fft.ifftn(regridded_data_echo_2_phase_corrected, norm='ortho')

#----------------------------------------------------------------------------------------
#--- AGR recon --------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

# create an array for the dual echo (multi_channel data)

signal = np.array([
    regridded_data_echo_1_phase_corrected,
    regridded_data_echo_2_phase_corrected
])
# we have to create the "coil" axis for the single coil
signal = np.expand_dims(signal.astype(np.complex128), 0)
# convert the complex signal into a real pseudo complex array
signal = np.ascontiguousarray(signal.view('(2,)float'))

# setup the readout inds for the apodized fft
k_fft = np.fft.fftfreq(ifft_echo_1.shape[0],
                       d=field_of_view_cm / ifft_echo_1.shape[0])
K0_fft, K1_fft, K2_fft = np.meshgrid(k_fft, k_fft, k_fft, indexing='ij')
K_abs_fft = np.sqrt(K0_fft**2 + K1_fft**2 + K2_fft**2)

readout_time_image_ms = np.full(K_abs_fft.shape, -1)

# setup a pseudo-complex kspace mask that indicates which elements of kspace where sampled
kmask = np.zeros(signal.shape)
for j in range(1):
    for i in range(2):
        kmask[j, i, ..., 0] = (K_abs_fft < abs_k_spoke.max()).astype(float)
        kmask[j, i, ..., 1] = (K_abs_fft < abs_k_spoke.max()).astype(float)

inds = np.where(kmask[0, 0, ..., 0])
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
kmax = 1 / (2 * field_of_view_cm / 64)
hmask = h_win(K_abs_fft.ravel() * 32 / kmax).reshape(K_abs_fft.shape)

ifft_echo_1_filtered = np.fft.ifftn(hmask *
                                    regridded_data_echo_1_phase_corrected,
                                    norm='ortho')
ifft_echo_2_filtered = np.fft.ifftn(hmask *
                                    regridded_data_echo_2_phase_corrected,
                                    norm='ortho')

Gam_bounds = (ifft_echo_1.shape[0]**3) * [(0.001, 1)]
Gam_recon = np.clip(
    np.abs(ifft_echo_2_filtered) / (np.abs(ifft_echo_1_filtered) + 0.001),
    0.001, 1)
Gam_recon[np.abs(ifft_echo_2_filtered) < 0.1 *
          np.abs(ifft_echo_2_filtered).max()] = 1
Gam_recon = gaussian_filter(Gam_recon, 1)

#vi3 = pv.ThreeAxisViewer(Gam_recon)

# setup the anatomical prior image and the nearest neighbors arrays for
# the Bowsher prior
aimg = np.load(t1_nifti_file)
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

ims = 6 * [{
    'cmap': plt.cm.Greys_r,
    'vmin': 0,
    'vmax': 2.5
}] + [{
    'cmap': plt.cm.Greys_r,
    'vmin': 0.0,
    'vmax': 1.
}]

vi4 = pv.ThreeAxisViewer([
    np.flip(abs_recons[-1, ...], (0, 1)),
    np.flip(np.abs(ifft_echo_1), (0, 1)),
    np.flip(np.abs(ifft_echo_2), (0, 1)),
    np.flip(np.abs(ifft_echo_1_filtered), (0, 1)),
    np.flip(np.abs(ifft_echo_2_filtered), (0, 1)),
    np.flip(3 * aimg / aimg.max(), (0, 1)),
    np.flip(Gam_recons[-1, ...], (0, 1))
],
                         imshow_kwargs=ims)
vi4.fig.savefig(output_dir / '00_screenshot.png')

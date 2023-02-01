# TODO: - understand sampling density difference in the center (factor of 2)

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from simulate_dual_echo_tpi_na_data import regrid_tpi_data
from pymirc.image_operations import zoom3d

import pymirc.viewer as pv

#-------------------------------------------------------------------------
# input parameters
gridded_data_matrix_size: int = 128
noise_level: float = 0  #4e5
gradient_file: str = '/data/sodium_mr/tpi_gradients/n28p4dt10g16_23Na_v1'
#gradient_file: str = '/data/sodium_mr/tpi_gradients/n28p4dt10g32_23Na_v0'

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# restore the saved noise-free nufft data from file

simulated_data_path = Path(
    '/data') / 'sodium_mr' / f'brainweb_{Path(gradient_file).name}'

data = np.load(simulated_data_path / 'simulated_nufft_data.npz')

nonuniform_data_long_echo_1 = data['nonuniform_data_long_echo_1']
nonuniform_data_long_echo_2 = data['nonuniform_data_long_echo_2']
nonuniform_data_short_echo_1 = data['nonuniform_data_short_echo_1']
nonuniform_data_short_echo_2 = data['nonuniform_data_short_echo_2']
k0 = data['k0']
k1 = data['k1']
k2 = data['k2']
kp = data['kp']
kmax = data['kmax']
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
print(kx[abs_k <= kmax].size)

#fig = plt.figure(figsize=(8, 8))
#i = 1500
#jmax = np.where(abs_k[i, :] <= kp)[0].max()
#ax = fig.add_subplot(1, 1, 1, projection='3d')
#ax.scatter3D(kx[i, :jmax], ky[i, :jmax], kz[i, :jmax], s=0.5)
#ax.scatter3D(kx[i, jmax:], ky[i, jmax:], kz[i, jmax:], s=0.5)
#ax.set_xlim(kx.min(), kx.max())
#ax.set_ylim(ky.min(), ky.max())
#ax.set_zlim(kz.min(), kz.max())
#fig.tight_layout()
#fig.show()

#fig, ax = plt.subplots()
#ax.plot(abs_k[-1, :])
#ax.axhline(kmax)
#ax.axhline(kp)
#fig.show()

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

cutoff = 1.
window = np.ones(100)

# allocated memory for output arrays
sampling_weights = np.zeros(
    (gridded_data_matrix_size, gridded_data_matrix_size,
     gridded_data_matrix_size),
    dtype=complex)

regridded_data_echo_1 = np.zeros(
    (gridded_data_matrix_size, gridded_data_matrix_size,
     gridded_data_matrix_size),
    dtype=complex)
regridded_data_echo_2 = np.zeros(
    (gridded_data_matrix_size, gridded_data_matrix_size,
     gridded_data_matrix_size),
    dtype=complex)

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

print('calculating weights')
regrid_tpi_data(gridded_data_matrix_size,
                1 / field_of_view_cm,
                nonuniform_data_long_echo_1,
                k0.size,
                k0.ravel(),
                k1.ravel(),
                k2.ravel(),
                kmax,
                kp,
                window,
                cutoff,
                sampling_weights,
                correct_tpi_sampling_density=True,
                output_weights=True)

print('regridding echo 1 data')
regrid_tpi_data(gridded_data_matrix_size,
                1 / field_of_view_cm,
                0.6 * noisy_nonuniform_data_short_echo_1 +
                0.4 * noisy_nonuniform_data_long_echo_1,
                k0.size,
                k0.ravel(),
                k1.ravel(),
                k2.ravel(),
                kmax,
                kp,
                window,
                cutoff,
                regridded_data_echo_1,
                correct_tpi_sampling_density=True,
                output_weights=False)

print('regridding echo 2 data')
regrid_tpi_data(gridded_data_matrix_size,
                1 / field_of_view_cm,
                0.6 * noisy_nonuniform_data_short_echo_2 +
                0.4 * noisy_nonuniform_data_long_echo_2,
                k0.size,
                k0.ravel(),
                k1.ravel(),
                k2.ravel(),
                kmax,
                kp,
                window,
                cutoff,
                regridded_data_echo_2,
                correct_tpi_sampling_density=True,
                output_weights=False)

print('IFFT recon')
# don't forget to fft shift the data since the regridding function puts the kspace
# origin in the center of the array
regridded_data_echo_1 = np.fft.fftshift(regridded_data_echo_1)
regridded_data_echo_2 = np.fft.fftshift(regridded_data_echo_2)

# numpy's fft handles the phase factor of the DFT diffrently compared to pynufft
# so we have to apply a phase factor to the regridded data
# in 1D this phase factor is [1,-1,1,-1, ...]
# in 3D it is the 3D checkerboard version of this
# see here for details https://stackoverflow.com/questions/24077913/discretized-continuous-fourier-transform-with-numpy
tmp_x = np.arange(gridded_data_matrix_size)
TMP_X, TMP_Y, TMP_Z = np.meshgrid(tmp_x, tmp_x, tmp_x)

phase_correction = ((-1)**TMP_X) * ((-1)**TMP_Y) * ((-1)**TMP_Z)
regridded_data_echo_1_phase_corrected = phase_correction * regridded_data_echo_1
regridded_data_echo_2_phase_corrected = phase_correction * regridded_data_echo_2

# IFFT of the regridded data
ifft_echo_1 = np.fft.ifftn(regridded_data_echo_1_phase_corrected)
ifft_echo_2 = np.fft.ifftn(regridded_data_echo_2_phase_corrected)

# the regridding in kspace uses trilinear interpolation (convolution with a triangle)
# we the have to divide by the FT of a triangle (sinc^2)
tmp_x = np.linspace(-0.5, 0.5, gridded_data_matrix_size)
TMP_X, TMP_Y, TMP_Z = np.meshgrid(tmp_x, tmp_x, tmp_x)

# corretion field is sinc(R)**2
corr_field = np.sinc(np.sqrt(TMP_X**2 + TMP_Y**2 + TMP_Z**2))**2

ifft_echo_1_corr = ifft_echo_1 / corr_field
ifft_echo_2_corr = ifft_echo_2 / corr_field

# interpolate magnitude images to simulation grid size (which can be different from gridded data size)
a = zoom3d(np.abs(ifft_echo_1_corr),
           simulation_matrix_size / gridded_data_matrix_size)
b = zoom3d(np.abs(ifft_echo_2_corr),
           simulation_matrix_size / gridded_data_matrix_size)

ims = [{}] + 2 * [{'vmin': 0, 'vmax': 400}]
vi = pv.ThreeAxisViewer([na_image, a, b], imshow_kwargs=ims)

vi2 = pv.ThreeAxisViewer(sampling_weights.real)

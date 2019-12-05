# small demo script to verify implementation of discrete FT (with FFT)

import numpy as np
from   numba import njit, prange
import matplotlib.pyplot as py
from   matplotlib.colors import LogNorm

from pymirc.image_operations import zoom3d

py.ion()
py.rc('image', cmap='gray')

#--------------------------------------------------------------
# this is the super naiv direct implementation of the DFT is 
# horribly slow. For the apodized FFT is more efficient
# to calculate multiple FFTs
@njit(parallel = True)
def dft2D(f, k0, k1):
  n0,n1 = f.shape
  F     = np.zeros((n0,n1), dtype = np.complex64)
  x0    = np.outer(np.arange(n0),np.ones(n1))
  x1    = np.outer(np.ones(n0),np.arange(n1))

  for i in prange(n0): 
    for j in range(n1): 
      F[i,j] = (f*np.exp(-2*np.pi*(k0[i]*x0 + k1[j]*x1)*1j)).sum()

  return F

#--------------------------------------------------------------

# load the brain web labels
data = np.load('54.npz')
t1     = data['arr_0']
labels = data['arr_1']
lab = np.pad(labels[:,:,132].transpose(), ((0,0),(36,36)),'constant')

# CSF = 1, GM = 2, WM = 3
csf_inds = np.where(lab == 1) 
gm_inds  = np.where(lab == 2)
wm_inds  = np.where(lab == 3)

# set up array for trans. magnetization
f = np.zeros(lab.shape)
f[csf_inds] = 1.1
f[gm_inds]  = 0.8
f[wm_inds]  = 0.7

# set up array for T2* times
T2star = np.ones(lab.shape)
T2star[csf_inds] = 48.
T2star[gm_inds]  = 12.
T2star[wm_inds]  = 15.

# regrid to a 256 grid
f      = zoom3d(np.expand_dims(f,-1),(256/434,256/434,1))[...,0]
T2star = zoom3d(np.expand_dims(T2star,-1),(256/434,256/434,1))[...,0]

# setup the frequency array as used in numpy fft
tmp    = np.fft.fftfreq(f.shape[0])
k0, k1 = np.meshgrid(tmp, tmp, indexing = 'ij')
abs_k  = np.sqrt(k0**2 + k1**2)

# generate array of k-space readout times
n_readout_bins     = 32
readout_ind_array  = (abs_k * (n_readout_bins**2) / abs_k.max()) // n_readout_bins
read_out_times     = 400*abs_k[readout_ind_array == (n_readout_bins-1)].mean() * np.linspace(0,1,n_readout_bins)
read_out_inds      = []

# generate the signal apodization images
apo_images = np.zeros((n_readout_bins,) + f.shape)

for i, t_read in enumerate(read_out_times):
  apo_images[i,...] = np.exp(-t_read / T2star)
  read_out_inds.append(np.where(readout_ind_array == i))

F = np.zeros(f.shape, dtype = np.complex)
for i in range(apo_images.shape[0]):
  tmp = np.fft.fft2(apo_images[i,...] * f)
  F[read_out_inds[i]] = tmp[read_out_inds[i]]

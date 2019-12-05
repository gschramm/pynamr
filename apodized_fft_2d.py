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
def apodized_fft_2d(f, readout_inds, apo_images):
  """ Calculate 2D apodized FFT of an image (e.g. caused by T2* decay during readout
  
  Parameters
  ----------

  f : 2d numpy array of shape (n0,n1)

  readout_inds : list of array indices (nr elements)
    containing the 2D array indicies that read out at every time point

  apo_images : 3d numpt array of shape(nr,n0n,1)
    containing the nultiplicative apodization images at each readout time point
  """

  F = np.zeros(f.shape, dtype = np.complex)

  for i in range(apo_images.shape[0]):
    tmp = np.fft.fft2(apo_images[i,...] * f)
    F[readout_inds[i]] = tmp[readout_inds[i]]

  return F

#--------------------------------------------------------------
def adjoint_apodized_fft_2d(F, readout_inds, apo_images):
  """ Calculate 2D apodized FFT of an image (e.g. caused by T2* decay during readout
  
  Parameters
  ----------

  F : 2d numpy array of shape (n0,n1)

  readout_inds : list of array indices (nr elements)
    containing the 2D array indicies that read out at every time point

  apo_images : 3d numpt array of shape(nr,n0n,1)
    containing the nultiplicative apodization images at each readout time point
  """

  n0, n1 = F.shape
  f      = np.zeros(F.shape, dtype = np.complex)

  for i in range(apo_images.shape[0]):
    tmp = np.zeros(f.shape, dtype = np.complex)
    tmp[readout_inds[i]] = F[readout_inds[i]]

    f += apo_images[i,...] * np.fft.ifft2(tmp)

  f *= (n0*n1)

  return f

#--------------------------------------------------------------

def apo_images(readout_times, T2star):
  apo_imgs = np.zeros((n_readout_bins,) + T2star.shape)

  for i, t_read in enumerate(readout_times):
    apo_imgs[i,...] = np.exp(-t_read / T2star)

  return apo_imgs

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
readout_times      = 400*abs_k[readout_ind_array == (n_readout_bins-1)].mean() * np.linspace(0,1,n_readout_bins)
readout_inds       = []

for i, t_read in enumerate(readout_times):
  readout_inds.append(np.where(readout_ind_array == i))

# generate the signal apodization images
apo_imgs  = apo_images(readout_times, T2star)

#----------------------------------------------------------
#--- simulate the signal

signal = apodized_fft_2d(f, readout_inds, apo_imgs)

# add noise to signal
noise_level = 0 # 1e0 
signal = signal + noise_level*(np.random.randn(256,256) + np.random.randn(256,256)*1j)

#----------------------------------------------------------
#--- do the recon

niter = 250
step  = 1.8/np.prod(f.shape)

recon        = np.fft.ifft2(signal)
T2star_recon = T2star.copy()
#T2star_recon = np.zeros(T2star.shape) + T2star.max()

apo_imgs_recon = apo_images(readout_times, T2star_recon)

recons = np.zeros((niter + 1,) + f.shape, dtype = np.complex)
recons[0,...] = recon

cost = np.zeros(niter)

for it in range(niter):
  exp_data = apodized_fft_2d(recon, readout_inds, apo_imgs_recon)
  diff     = exp_data - signal
  recon    = recon - step*adjoint_apodized_fft_2d(diff, readout_inds, apo_imgs_recon)
  recons[it + 1, ...] = recon
  cost[it] = 0.5*(diff*diff.conj()).sum().real
  print(it + 1, niter, round(cost[it],4))


#----------------------------------------------------------
#--- plot the results

vmax = min(1.5*f.max(),max(f.max(),np.abs(recon).max()))

fig, ax = py.subplots(3,3,figsize = (9,9))
ax[0,0].imshow(f, vmin = 0, vmax = vmax)
ax[0,1].imshow(np.abs(recons[0,:]), vmin = 0, vmax = vmax)
ax[0,2].imshow(np.abs(recon), vmin = 0, vmax = vmax)
ax[1,0].imshow(np.abs(recon) - f, vmin = -0.1*vmax, vmax = 0.1*vmax, cmap = py.cm.bwr)
ax[1,1].plot(f[:,128],'k')
ax[1,1].plot(np.abs(recons[0,...])[:,128],'b:')
ax[1,1].plot(np.abs(recon)[:,128],'r:')
ax[1,2].plot(f[128,:],'k')
ax[1,2].plot(np.abs(recons[0,...])[128,:],'b:')
ax[1,2].plot(np.abs(recon)[128,:],'r:')
ax[2,0].imshow(T2star, vmin = 0, vmax = T2star.max())
ax[2,1].imshow(T2star_recon, vmin = 0, vmax = T2star.max())

ax[0,0].set_title('ground truth')
ax[0,1].set_title('ifft of signal')
ax[0,2].set_title('iterative recon')
ax[2,0].set_title('ground truth T2*')
ax[2,1].set_title('recon T2*')

fig.tight_layout()
fig.show()



##----------------------------------------------------------
##--- adjoint test
#n = 256
#f = np.random.rand(n,n) + np.random.rand(n,n)*1j
#F = np.random.rand(n,n) + np.random.rand(n,n)*1j
#
#f_fwd  = apodized_fft_2d(f, readout_inds, apo_imgs)
#F_back = adjoint_apodized_fft_2d(F, readout_inds, apo_imgs)
#
#print((f_fwd * F.conj()).sum())
#print((f * F_back.conj()).sum())

#----------------------------------------------------------
#--- power iterations : largest eigenvalue is n0*n1
#b = f.copy()
#for it in range(25):
#  b_fwd = apodized_fft_2d(b, readout_inds, apo_imgs)
#  norm  = np.sqrt((b_fwd * b_fwd.conj()).sum().real)
#  b     = b_fwd / norm
#  print(norm)


#F = apodized_fft_2d(f, readout_inds, apo_imgs)
#r = np.fft.ifft2(F)



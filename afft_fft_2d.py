import numpy as np
import matplotlib.pyplot as py
from   matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

from pymirc.image_operations import zoom3d, complex_grad, complex_div

from prox import prox_tv, prox_pls

py.ion()
py.rc('image', cmap='gray')

#--------------------------------------------------------------
def apodized_fft_2d(f, readout_inds, apo_images):
  """ Calculate 2D apodized FFT of an image (e.g. caused by T2* decay during readout
  
  Parameters
  ----------

  f : a float64 2d numpy array of shape (n0,n1,2)
    [...,0] is considered as the real part
    [...,1] is considered as the imag part

  readout_inds : list of array indices (nr elements)
    containing the 2D array indicies that read out at every time point

  apo_images : 3d numpy array of shape(nr,n0,n1)
    containing the nultiplicative apodization images at each readout time point
  """

  # create a complex view of the input real input array with two channels
  f_comp = f.view(dtype=np.complex128)[...,0]

  F_comp = np.zeros(f_comp.shape, dtype = np.complex128)

  for i in range(apo_images.shape[0]):
    tmp = np.fft.fft2(apo_images[i,...] * f_comp)
    F_comp[readout_inds[i]] = tmp[readout_inds[i]]

  # we normalize to get the norm of the operator to the norm of the gradient op
  F_comp *= np.sqrt(4*f_comp.ndim) / np.sqrt(np.prod(f_comp.shape))

  return F_comp

#--------------------------------------------------------------
def adjoint_apodized_fft_2d(F_comp, readout_inds, apo_images):
  """ Calculate 2D apodized FFT of an image (e.g. caused by T2* decay during readout
  
  Parameters
  ----------

  F : a complex128 2d numpy array of shape (n0,n1)

  readout_inds : list of array indices (nr elements)
    containing the 2D array indicies that read out at every time point

  apo_images : 3d numpt array of shape(nr,n0,n1)
    containing the nultiplicative apodization images at each readout time point
  """

  n0, n1 = F_comp.shape
  f_comp = np.zeros(F_comp.shape, dtype = np.complex128)

  for i in range(apo_images.shape[0]):
    tmp = np.zeros(f_comp.shape, dtype = np.complex128)
    tmp[readout_inds[i]] = F_comp[readout_inds[i]]

    f_comp += apo_images[i,...] * np.fft.ifft2(tmp)

  f_comp *=  (np.sqrt(np.prod(F_comp.shape)) * np.sqrt(4*F_comp.ndim))

  f = f_comp.view(dtype=np.float64).reshape(f_comp.shape + (2,))

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
ff = np.zeros(lab.shape)
ff[csf_inds] = 1.1
ff[gm_inds]  = 0.8
ff[wm_inds]  = 0.7

# set up array for T2* times
T2star = np.ones(lab.shape)
T2star[csf_inds] = 48.
T2star[gm_inds]  = 12.
T2star[wm_inds]  = 15.

# regrid to a 256 grid
ff     = zoom3d(np.expand_dims(ff,-1),(256/434,256/434,1))[...,0]
T2star = zoom3d(np.expand_dims(T2star,-1),(256/434,256/434,1))[...,0]

# add empty imaginary channel to f
f = np.zeros(ff.shape + (2,))
f[...,0] = ff
f[...,1] = 0

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
noise_level = 0 # 1e-2
signal = signal + noise_level*(np.random.randn(256,256) + np.random.randn(256,256)*1j)


#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
#--- do the recon

alg   = 'pdhg'
niter = 500
lam   = 1e-2
prior = 'PLS2'

tmp   = (np.fft.ifft2(signal) * np.sqrt(np.prod(f.shape[:-1])) / np.sqrt(4*signal.ndim))
recon = tmp.view(dtype=np.float64).reshape(tmp.shape + (2,))
init_recon = recon.copy()

T2star_recon = T2star.copy()
#T2star_recon = np.zeros(T2star.shape) + T2star.max()

apo_imgs_recon = apo_images(readout_times, T2star_recon)

xi = np.zeros((2*recon[...,0].ndim,) + recon.shape[:-1])
complex_grad(-2*f, xi)

#recons = np.zeros((niter + 1,) + recon.shape)
#recons[0,...] = recon

cost  = np.zeros(niter)
cost1 = np.zeros(niter)
cost2 = np.zeros(niter)

##----------------------------------------------------------
##--- power iterations to get norm of MR operator
#b = np.random.random(recon.shape)
#for it in range(50):
#  b_fwd = apodized_fft_2d(b, readout_inds, apo_imgs)
#  L     = np.sqrt((b_fwd * b_fwd.conj()).sum().real)
#  b     = b_fwd.view(dtype=np.float64).reshape(b_fwd.shape + (2,)) / L

#----------------------------------------------------------
if alg == 'landweber':
  step  = 0.2

  for it in range(niter):
    exp_data = apodized_fft_2d(recon, readout_inds, apo_imgs_recon)
    diff     = exp_data - signal
    recon    = recon - step*adjoint_apodized_fft_2d(diff, readout_inds, apo_imgs_recon)
    #recons[it + 1, ...] = recon
    cost[it] = 0.5*(diff*diff.conj()).sum().real
    print(it + 1, niter, round(cost[it],4))

#----------------------------------------------------------
elif alg == 'pdhg':
  L     = np.sqrt(2*recon.ndim*4)
  sigma = (1e0)/L
  tau   = 1./(sigma*(L**2))

  recon_bar  = recon.copy()
  recon_dual = np.zeros(signal.shape, dtype = signal.dtype)

  grad_dual  = np.zeros((2*recon[...,0].ndim,) + recon.shape[:-1])

  for it in range(niter):
    diff        = apodized_fft_2d(recon_bar, readout_inds, apo_imgs_recon) - signal
    recon_dual += sigma * diff / (1 + sigma*lam)
    recon_old   = recon.copy()

    tmp  = np.zeros((2*recon[...,0].ndim,) + recon.shape[:-1])
    complex_grad(recon_bar, tmp)
    grad_dual += sigma * tmp
    if prior == 'TV':
      prox_tv(grad_dual, 1.)
    elif prior == 'PLS1':
      prox_pls(grad_dual, xi, 1., 1)
    elif prior == 'PLS2':
      prox_pls(grad_dual, xi, 1., 2)

    recon += tau*(complex_div(grad_dual) - adjoint_apodized_fft_2d(recon_dual, readout_inds, apo_imgs_recon))
 
    theta      = 1.
    recon_bar  = recon + theta*(recon - recon_old)

    #recons[it + 1, ...] = recon

    # calculate the cost
    tmp  = np.zeros((2*recon[...,0].ndim,) + recon.shape[:-1])
    complex_grad(recon, tmp)
    tmp2 = apodized_fft_2d(recon, readout_inds, apo_imgs_recon) - signal
    cost[it]  = (0.5/lam)*(tmp2*tmp2.conj()).sum().real + np.linalg.norm(tmp, axis=0).sum()
    cost1[it] = (0.5/lam)*(tmp2*tmp2.conj()).sum().real
    cost2[it] = np.linalg.norm(tmp, axis=0).sum()
    print(it + 1, niter, round(cost1[it],4), round(cost2[it],4), round(cost[it],4))

#----------------------------------------------------------
# make some plots

fig, ax = py.subplots(1,3, figsize = (12,4))
ax[0].imshow(np.linalg.norm(init_recon, axis=-1), vmax = 1.1*f.max())
ax[1].imshow(np.linalg.norm(recon, axis=-1), vmax = 1.1*f.max())
ax[2].imshow(np.linalg.norm(f, axis=-1), vmax = 1.1*f.max())
fig.tight_layout()
fig.show()




##----------------------------------------------------------
##--- adjoint test
#n = 256
#f = np.random.rand(n,n,2)
#F = np.zeros((n,n), dtype = np.complex128)
#F.real = np.random.rand(n,n)
#F.imag = np.random.rand(n,n)
#
#f_fwd  = apodized_fft_2d(f, readout_inds, apo_imgs)
#F_back = adjoint_apodized_fft_2d(F, readout_inds, apo_imgs)
#
#print((f_fwd*F.conj()).sum().real)
#print((f*F_back).sum())

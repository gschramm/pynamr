# small demo script to verify implementation of discrete FT (with FFT)
# TODO: correct k-space sampling (not distance from center ...)

import numpy as np
from scipy.optimize import fmin_l_bfgs_b, fmin_cg

import matplotlib.pyplot as py
from   matplotlib.colors import LogNorm

from apodized_fft      import *
from nearest_neighbors import *
from bowsher           import *

from scipy.ndimage import zoom
from pymirc.image_operations import complex_grad, complex_div

#--------------------------------------------------------------
def mr_data_fidelity(recon, signal, readout_inds, apo_imgs):

  exp_data = apodized_fft(recon, readout_inds, apo_imgs)
  diff     = exp_data - signal
  cost     = 0.5*(diff**2).sum()

  return cost

#--------------------------------------------------------------
def mr_data_fidelity_grad(recon, signal, readout_inds, apo_imgs):

  exp_data = apodized_fft(recon, readout_inds, apo_imgs_recon)
  diff     = exp_data - signal
  grad     = adjoint_apodized_fft(diff, readout_inds, apo_imgs_recon)

  return grad

#--------------------------------------------------------------------
def mr_bowsher_cost(recon, recon_shape, signal, readout_inds, apo_imgs, beta, ninds, ninds2, method):
  # ninds2 is a dummy argument to have the same arguments for
  # cost and its gradient

  isflat = False
  if recon.ndim == 1:  
    isflat = True
    recon  = recon.reshape(recon_shape)

  cost = mr_data_fidelity(recon, signal, readout_inds, apo_imgs)

  if beta > 0:
    cost += beta*bowsher_prior_cost(recon[...,0], ninds, method)
    cost += beta*bowsher_prior_cost(recon[...,1], ninds, method)

  if isflat:
    recon = recon.flatten()
   
  return cost

#--------------------------------------------------------------------
def mr_bowsher_grad(recon, recon_shape, signal, readout_inds, apo_imgs, beta, ninds, ninds2, method):

  isflat = False
  if recon.ndim == 1:  
    isflat = True
    recon  = recon.reshape(recon_shape)

  grad = mr_data_fidelity_grad(recon, signal, readout_inds, apo_imgs)

  if beta > 0:

    grad[...,0] += beta*bowsher_prior_grad(recon[...,0], ninds, ninds2, method)
    grad[...,1] += beta*bowsher_prior_grad(recon[...,1], ninds, ninds2, method)

  if isflat:
    recon = recon.flatten()
    grad  = grad.flatten()

  return grad


#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
n = 128

alg    = 'lbfgs'
niter  = 20
beta   = 1e-1
method = 0

time_fac = 3.

T2short = -1   # -1 -> inverse crime, float -> constant value
T2long  = -1   # -1 -> inverse crime, float -> constant value

#--------------------------------------------------------------------------------------

py.ion()
py.rc('image', cmap='gray')

# load the brain web labels
data = np.load('54.npz')
t1     = data['arr_0']
labels = data['arr_1']
lab    = np.pad(labels[:,:,132].transpose(), ((0,0),(36,36)),'constant')

#===========================================================================================

# CSF = 1, GM = 2, WM = 3
csf_inds = np.where(lab == 1) 
gm_inds  = np.where(lab == 2)
wm_inds  = np.where(lab == 3)

# set up array for trans. magnetization
f = np.zeros(lab.shape)
f[csf_inds] = 1.1
f[gm_inds]  = 0.8
f[wm_inds]  = 0.7

# regrid to a 256 grid
f          = zoom(np.expand_dims(f,-1),(n/434,n/434,1), order = 1, prefilter = False)[...,0]
lab_regrid = zoom(lab, (n/434,n/434), order = 0, prefilter = False) 

# set up array for T2* times
T2star_short = np.zeros((n,n)) + 1e5
T2star_short[lab_regrid == 1] = 50.
T2star_short[lab_regrid == 2] = 3.
T2star_short[lab_regrid == 3] = 3.

T2star_long = np.zeros((n,n)) + 1e5
T2star_long[lab_regrid == 1] = 50.
T2star_long[lab_regrid == 2] = 15.
T2star_long[lab_regrid == 3] = 15.


#===========================================================================================

# add imag part to
f = np.stack((f,np.zeros(f.shape)), axis = -1)

# setup the frequency array as used in numpy fft
tmp    = np.fft.fftfreq(f.shape[0])
k0, k1 = np.meshgrid(tmp, tmp, indexing = 'ij')
abs_k  = np.sqrt(k0**2 + k1**2)

# generate array of k-space readout times
# this is contains the readout time as a function of |k|
n_readout_bins = 100
tmp            = time_fac*np.loadtxt('readout_times.csv', delimiter = ',')
readout_times  = np.interp(np.linspace(0,tmp.max(),n_readout_bins), np.linspace(0,tmp.max(),len(tmp)), tmp)
readout_ind_array  = (abs_k * (n_readout_bins**2) / abs_k.max()) // n_readout_bins

readout_inds = []

for i, t_read in enumerate(readout_times):
  readout_inds.append(np.where(readout_ind_array == i))

# generate the signal apodization images
apo_imgs  = apo_images(readout_times, T2star_short, T2star_long)

#----------------------------------------------------------
#--- simulate the signal

signal = apodized_fft(f, readout_inds, apo_imgs)

# add noise to signal
noise_level = 0 # 1e0 
signal += noise_level*(np.random.randn(*signal.shape))

#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
#--- do the recon

if T2short == -1:
  T2star_short_recon = T2star_short.copy()
else :
  T2star_short_recon = np.zeros((n,n)) + T2short

if T2long == -1:
  T2star_long_recon = T2star_long.copy()
else :
  T2star_long_recon = np.zeros((n,n)) + T2long


apo_imgs_recon = apo_images(readout_times, T2star_short_recon, T2star_long_recon)

init_recon  = np.fft.ifft2(np.squeeze(signal.view(dtype = np.complex128))) * np.sqrt(np.prod(f.shape)) / np.sqrt(4*signal.ndim)

init_recon  = init_recon.view('(2,)float')

# --- set up stuff for the prior
aimg  = f.max() - f[...,0]
aimg += 0.001*aimg.max()*np.random.random(aimg.shape)

# beta = 1e-4 reasonable for inverse crime
s    = np.array([[1,1,1], 
                 [1,0,1], 
                 [1,1,1]])
nnearest = 3 

ninds  = np.zeros((np.prod(aimg.shape),nnearest), dtype = np.uint32)
nearest_neighbors(aimg,s,nnearest,ninds)
ninds2 = is_nearest_neighbor_of(ninds)

cost = []

cb = lambda x: cost.append(mr_bowsher_cost(x, recon_shape, signal, readout_inds, 
                           apo_imgs_recon, beta, ninds, ninds2, method))

if alg == 'lbfgs' or alg == 'cg':
  print('starting bfgs')
  recon       = init_recon.copy()
  recon_shape = init_recon.shape
  recon       = recon.flatten()

  if alg == 'lbfgs':
    res = fmin_l_bfgs_b(mr_bowsher_cost,
                        recon, 
                        fprime = mr_bowsher_grad, 
                        args = (recon_shape, signal, readout_inds, apo_imgs_recon, beta, ninds, ninds2, method), 
                        callback = cb,
                        maxiter = niter, 
                        disp = 1)
    recon = res[0].reshape(recon_shape)
  elif alg == 'cg':
    res = fmin_cg(mr_bowsher_cost,
                  recon, 
                  fprime = mr_bowsher_grad, 
                  args = (recon_shape, signal, readout_inds, apo_imgs_recon, beta, ninds, ninds2, method), 
                  callback = cb,
                  maxiter = niter, 
                  disp = 1)

    recon = res.reshape(recon_shape)

#----------------------------------------------------------
#--- plot the results

vmax = 1.2*f.max()

abs_f           = np.linalg.norm(f,axis=-1)
abs_init_recon  = np.linalg.norm(init_recon,axis=-1)
abs_init_recon *= abs_f.sum() / abs_init_recon.sum()

abs_recon      = np.linalg.norm(recon,axis=-1)

fig, ax = py.subplots(3,4,figsize = (12,9))
ax[0,0].imshow(abs_f, vmin = 0, vmax = vmax)
ax[0,1].imshow(abs_init_recon, vmin = 0, vmax = vmax)
ax[0,2].imshow(abs_recon, vmin = 0, vmax = vmax)
ax[0,3].plot(readout_times)

ax[1,0].imshow(abs_recon - abs_f, vmin = -0.1*vmax, vmax = 0.1*vmax, cmap = py.cm.bwr)
ax[1,1].plot(abs_f[:,n//2],'k')
ax[1,1].plot(abs_init_recon[:,n//2],'b:')
ax[1,1].plot(abs_recon[:,n//2],'r:')
ax[1,2].plot(abs_f[n//2,:],'k')
ax[1,2].plot(abs_init_recon[n//2,:],'b:')
ax[1,2].plot(abs_recon[n//2,:],'r:')
ax[1,3].loglog(np.arange(1,len(cost)+1), cost)

ax[2,0].imshow(T2star_short,       vmin = 0,  vmax = 15)
ax[2,1].imshow(T2star_long,        vmin = 0,  vmax = 65)
ax[2,2].imshow(T2star_short_recon, vmin = 0,  vmax = 15)
ax[2,3].imshow(T2star_long_recon,  vmin = 0,  vmax = 65)

ax[0,0].set_title('ground truth')
ax[0,1].set_title('init recon (ifft)')
ax[0,2].set_title('iterative recon')
ax[0,3].set_title('readout times (|k|)')

ax[1,3].set_title('cost')

ax[2,0].set_title('gt short T2*')
ax[2,1].set_title('gt long T2*')
ax[2,2].set_title('recon short T2*')
ax[2,3].set_title('recon long T2*')

fig.tight_layout()
fig.show()




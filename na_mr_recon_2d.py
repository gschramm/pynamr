# small demo script to verify implementation of discrete FT (with FFT)
# TODO: correct k-space sampling (not distance from center ...)

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

import matplotlib.pyplot as py
from   matplotlib.colors import LogNorm

from apodized_fft_2d   import *
from nearest_neighbors import *
from bowsher           import *

from pymirc.image_operations import zoom3d, complex_grad, complex_div


py.ion()
py.rc('image', cmap='gray')

#--------------------------------------------------------------
def complex2d_to_flat_real(x):
  return x.view('(2,)float').flatten()

#--------------------------------------------------------------
def flat_real_to_complex2d(x):
  tmp = int(np.sqrt(x.shape[0]/2))
  y = x.reshape(tmp,tmp,2)

  return np.squeeze(y.view('complex'))

#--------------------------------------------------------------
def apo_images(readout_times, T2star_short, Tstar_long, C_short = 0.6, C_long = 0.4):
  apo_imgs = np.zeros((n_readout_bins,) + T2star_short.shape)

  for i, t_read in enumerate(readout_times):
    apo_imgs[i,...] = C_short*np.exp(-t_read / T2star_short) + C_long*np.exp(-t_read / T2star_long)

  return apo_imgs

#--------------------------------------------------------------
def mr_data_fidelity(recon, readout_inds, apo_imgs):

  isflat = False
  if recon.ndim == 1:  
    isflat = True
    recon  = flat_real_to_complex2d(recon)

  exp_data = apodized_fft_2d(recon, readout_inds, apo_imgs)
  diff     = exp_data - signal

  cost = 0.5*(diff*diff.conj()).sum().real

  if isflat:
    recon  = complex2d_to_flat_real(recon)

  return cost

#--------------------------------------------------------------
def mr_data_fidelity_grad(recon, readout_inds, apo_imgs):

  isflat = False
  if recon.ndim == 1:  
    isflat = True
    recon  = flat_real_to_complex2d(recon)

  exp_data = apodized_fft_2d(recon, readout_inds, apo_imgs_recon)
  diff     = exp_data - signal
  grad     = adjoint_apodized_fft_2d(diff, readout_inds, apo_imgs_recon)

  if isflat:
    grad   = complex2d_to_flat_real(grad)
    recon  = complex2d_to_flat_real(recon)

  return grad

#--------------------------------------------------------------------
def mr_bowsher_cost(recon, readout_inds, apo_imgs, beta, ninds, ninds2, method):
  # ninds2 is a dummy argument to have the same arguments for
  # cost and its gradient

  cost = mr_data_fidelity(recon, readout_inds, apo_imgs)

  if beta > 0:
    isflat = False
    if recon.ndim == 1:  
      isflat = True
      recon  = flat_real_to_complex2d(recon)

    cost += beta*bowsher_prior_cost(recon.real, ninds, method)
    cost += beta*bowsher_prior_cost(recon.imag, ninds, method)

    if isflat:
      recon  = complex2d_to_flat_real(recon)
   
  return cost

#--------------------------------------------------------------------
def mr_bowsher_grad(recon, readout_inds, apo_imgs, beta, ninds, ninds2, method):

  grad = mr_data_fidelity_grad(recon, readout_inds, apo_imgs)

  if beta > 0:
    isflat = False
    if recon.ndim == 1:  
      isflat = True
      recon  = flat_real_to_complex2d(recon)

    prior_grad      = np.zeros(recon.shape, dtype = np.complex128)
    prior_grad.real = bowsher_prior_grad(recon.real, ninds, ninds2, method)
    prior_grad.imag = bowsher_prior_grad(recon.imag, ninds, ninds2, method)

    if isflat:
      prior_grad  = complex2d_to_flat_real(prior_grad)
      recon = complex2d_to_flat_real(recon)

    grad += beta*prior_grad

  return grad

#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------

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
T2star_short = np.ones(lab.shape)
T2star_short[csf_inds] = 50.
T2star_short[gm_inds]  = 8.
T2star_short[wm_inds]  = 8.

T2star_long = np.ones(lab.shape)
T2star_long[csf_inds] = 50.
T2star_long[gm_inds]  = 15.
T2star_long[wm_inds]  = 15.

# regrid to a 256 grid
f            = zoom3d(np.expand_dims(f,-1),(256/434,256/434,1))[...,0]
T2star_short = zoom3d(np.expand_dims(T2star_short,-1),(256/434,256/434,1))[...,0]
T2star_long  = zoom3d(np.expand_dims(T2star_long,-1),(256/434,256/434,1))[...,0]

# setup the frequency array as used in numpy fft
tmp    = np.fft.fftfreq(f.shape[0])
k0, k1 = np.meshgrid(tmp, tmp, indexing = 'ij')
abs_k  = np.sqrt(k0**2 + k1**2)

# generate array of k-space readout times
n_readout_bins     = 50
readout_ind_array  = (abs_k * (n_readout_bins**2) / abs_k.max()) // n_readout_bins
readout_times      = 100*abs_k[readout_ind_array == (n_readout_bins-1)].mean() * np.linspace(0,1,n_readout_bins)
readout_inds       = []

for i, t_read in enumerate(readout_times):
  readout_inds.append(np.where(readout_ind_array == i))

# generate the signal apodization images
apo_imgs  = apo_images(readout_times, T2star_short, T2star_long)

#----------------------------------------------------------
#--- simulate the signal

signal = apodized_fft_2d(f, readout_inds, apo_imgs)

# add noise to signal
noise_level = 0 # 1e0 
signal = signal + noise_level*(np.random.randn(256,256) + np.random.randn(256,256)*1j)

#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
#--- do the recon

alg    = 'lbfgs'
niter  = 100
beta   = 1e1
method = 0

T2star_short_recon = T2star_short.copy()
T2star_long_recon  = T2star_long.copy()

apo_imgs_recon = apo_images(readout_times, T2star_short_recon, T2star_long_recon)

init_recon  = np.fft.ifft2(signal) * np.sqrt(np.prod(f.shape)) / np.sqrt(4*signal.ndim)

# --- set up stuff for the prior
aimg  = (f.max() - f.copy())
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

cb = lambda x: cost.append(mr_bowsher_cost(x,readout_inds, apo_imgs_recon, beta, ninds, ninds2, method))

if alg == 'lbfgs':
  print('starting bfgs')
  recon = complex2d_to_flat_real(init_recon)
  recon = np.ones(recon.shape)

  res = fmin_l_bfgs_b(mr_bowsher_cost,
                      recon, 
                      fprime = mr_bowsher_grad, 
                      args = (readout_inds, apo_imgs_recon, beta, ninds, ninds2, method), 
                      callback = cb,
                      maxfun = niter, 
                      maxiter = niter, 
                      disp = 1)

  recon = flat_real_to_complex2d(res[0])


##----------------------------------------------------------
##--- power iterations to get norm of MR operator
#n0, n1 = recon.shape
#b = np.random.rand(n0,n1) + np.random.rand(n0,n1)*1j
#for it in range(25):
#  b_fwd = apodized_fft_2d(b, readout_inds, apo_imgs)
#  L     = np.sqrt((b_fwd * b_fwd.conj()).sum().real)
#  b     = b_fwd / L
#  print(L)
##----------------------------------------------------------
#
#elif alg == 'pdhg':
#  L     = np.sqrt(2*recon.ndim*4)
#  sigma = (1e1)/L
#  tau   = 1./(sigma*(L**2))
#
#  # convexity parameter of data fidelity F*
#  gam = lam
# 
#  recon_bar  = recon.copy()
#  recon_dual = np.zeros(signal.shape, dtype = signal.dtype)
#
#  grad_dual  = np.zeros((2*recon.ndim,) + recon.shape)
#
#  for it in range(niter):
#    diff        = apodized_fft_2d(recon_bar, readout_inds, apo_imgs_recon) - signal
#    recon_dual += sigma * diff / (1 + sigma*lam)
#    recon_old   = recon.copy()
#
#    # forward step for complex gradient
#    tmp = np.zeros((2*recon.ndim,) + recon.shape)
#    complex_grad(recon_bar, tmp)
#    grad_dual += sigma * tmp
#    prox_tv(grad_dual, 1.)
#
#    #recon += tau*(-1*adjoint_apodized_fft_2d(recon_dual, readout_inds, apo_imgs_recon))
#    recon += tau*(complex_div(grad_dual) - adjoint_apodized_fft_2d(recon_dual, readout_inds, apo_imgs_recon))
# 
#    # update step sizes
#    theta  = 1.
#    #theta  = 1 / np.sqrt(1 + 2*gam*tau)
#    #tau   *= theta
#    #sigma /= theta
#
#    recon_bar   = recon + theta*(recon - recon_old)
#
#    # calculate the cost
#    tmp = np.zeros((2*recon.ndim,) + recon.shape)
#    complex_grad(recon, tmp)
#    tmp2 = apodized_fft_2d(recon, readout_inds, apo_imgs_recon) - signal
#    cost[it] = (0.5/lam)*(tmp2*tmp2.conj()).sum().real + np.linalg.norm(tmp, axis=0).sum()
#    cost1[it] = (tmp2*tmp2.conj()).sum().real
#    cost2[it] = np.linalg.norm(tmp, axis=0).sum()
#    print(it + 1, niter, round(cost1[it],4), round(cost2[it],4), round(cost[it],4))
#
#    recons[it + 1, ...] = recon
#



#----------------------------------------------------------
#--- plot the results

vmax = min(1.5*f.max(),max(f.max(),np.abs(recon).max()))

fig, ax = py.subplots(3,3,figsize = (9,9))
ax[0,0].imshow(f, vmin = 0, vmax = vmax)
ax[0,1].imshow(np.abs(init_recon), vmin = 0, vmax = vmax)
ax[0,2].imshow(np.abs(recon), vmin = 0, vmax = vmax)
ax[1,0].imshow(np.abs(recon) - f, vmin = -0.1*vmax, vmax = 0.1*vmax, cmap = py.cm.bwr)
ax[1,1].plot(f[:,128],'k')
ax[1,1].plot(np.abs(init_recon)[:,128],'b:')
ax[1,1].plot(np.abs(recon)[:,128],'r:')
ax[1,2].plot(f[128,:],'k')
ax[1,2].plot(np.abs(init_recon)[128,:],'b:')
ax[1,2].plot(np.abs(recon)[128,:],'r:')
ax[2,0].imshow(T2star_long, vmin = 0, vmax = T2star_long.max())
ax[2,1].imshow(T2star_long_recon, vmin = 0, vmax = T2star_long_recon.max())
ax[2,2].semilogy(np.arange(1,len(cost)+1), cost)

ax[0,0].set_title('ground truth')
ax[0,1].set_title('init recon')
ax[0,2].set_title('iterative recon')
ax[2,0].set_title('ground truth T2*')
ax[2,1].set_title('recon T2*')
ax[2,2].set_title('cost')

fig.tight_layout()
fig.show()




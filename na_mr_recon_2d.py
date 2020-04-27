# small demo script to verify implementation of discrete FT (with FFT)

import h5py
import os
import numpy as np
from scipy.optimize import fmin_l_bfgs_b, fmin_cg

import matplotlib.pyplot as py
from   matplotlib.colors import LogNorm

from apodized_fft      import *
from nearest_neighbors import *
from bowsher           import *
from readout_time      import readout_time

from scipy.ndimage import zoom, gaussian_filter

#--------------------------------------------------------------
def mr_data_fidelity(recon, signal, readout_inds, apo_imgs, kmask):

  exp_data = apodized_fft(recon, readout_inds, apo_imgs)*kmask
  diff     = exp_data - signal
  cost     = 0.5*(diff**2).sum()

  return cost

#--------------------------------------------------------------
def mr_data_fidelity_grad(recon, signal, readout_inds, apo_imgs, kmask):

  exp_data = apodized_fft(recon, readout_inds, apo_imgs_recon)*kmask
  diff     = exp_data - signal
  grad     = adjoint_apodized_fft(diff*kmask, readout_inds, apo_imgs_recon)

  return grad

#--------------------------------------------------------------------
def mr_bowsher_cost(recon, recon_shape, signal, readout_inds, apo_imgs, 
                    beta, ninds, ninds2, method, kmask):
  # ninds2 is a dummy argument to have the same arguments for
  # cost and its gradient

  isflat = False
  if recon.ndim == 1:  
    isflat = True
    recon  = recon.reshape(recon_shape)

  cost = mr_data_fidelity(recon, signal, readout_inds, apo_imgs, kmask)

  if beta > 0:
    cost += beta*bowsher_prior_cost(recon[...,0], ninds, method)
    cost += beta*bowsher_prior_cost(recon[...,1], ninds, method)

  if isflat:
    recon = recon.flatten()
   
  return cost

#--------------------------------------------------------------------
def mr_bowsher_grad(recon, recon_shape, signal, readout_inds, apo_imgs, 
                    beta, ninds, ninds2, method, kmask):

  isflat = False
  if recon.ndim == 1:  
    isflat = True
    recon  = recon.reshape(recon_shape)

  grad = mr_data_fidelity_grad(recon, signal, readout_inds, apo_imgs, kmask)

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
# parse the command line
from argparse import ArgumentParser

parser = ArgumentParser(description = '2D na mr simulation')

parser.add_argument('--T2star_recon_short',  default =  8,   type = float)
parser.add_argument('--T2star_recon_long',   default = 15,   type = float)
parser.add_argument('--beta',                default =  2,   type = float)
parser.add_argument('--sm_fwhm',             default = 1.5,  type = float)
parser.add_argument('--noise_level',         default = 0.2,  type = float)

parser.add_argument('--n',           default = 128,  type = int)
parser.add_argument('--niter',       default =  50,  type = int)
parser.add_argument('--method',      default =   0,  type = int, choices = [0,1])

parser.add_argument('--T2star_csf_short', default = 50, type = float)
parser.add_argument('--T2star_csf_long',  default = 50, type = float)
parser.add_argument('--T2star_gm_short',  default = 8,  type = float)
parser.add_argument('--T2star_gm_long',   default = 15, type = float)
parser.add_argument('--T2star_wm_short',  default = 9,  type = float)
parser.add_argument('--T2star_wm_long',   default = 18, type = float)

args = parser.parse_args()

# input parameters
n = args.n

niter       = args.niter
beta        = args.beta
sm_fwhm     = args.sm_fwhm
method      = args.method
noise_level = args.noise_level

T2star_csf_short = args.T2star_csf_short
T2star_csf_long  = args.T2star_csf_long 
T2star_gm_short  = args.T2star_gm_short 
T2star_gm_long   = args.T2star_gm_long  
T2star_wm_short  = args.T2star_wm_short 
T2star_wm_long   = args.T2star_wm_long  

save_recons  = True
add_B0_inhom = True

T2star_recon_short = args.T2star_recon_short   # -1 -> inverse crime, float -> constant value
T2star_recon_long  = args.T2star_recon_long    # -1 -> inverse crime, float -> constant value

#--------------------------------------------------------------------------------------

py.rc('image', cmap='gray')

# load the brain web labels
data = np.load('./data/54.npz')
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
T2star_short = np.full((n,n), T2star_csf_short)
T2star_short[lab_regrid == 1] = T2star_csf_short
T2star_short[lab_regrid == 2] = T2star_gm_short
T2star_short[lab_regrid == 3] = T2star_wm_short

T2star_long = np.full((n,n), T2star_csf_long)
T2star_long[lab_regrid == 1] = T2star_csf_long 
T2star_long[lab_regrid == 2] = T2star_gm_long
T2star_long[lab_regrid == 3] = T2star_wm_long

# add a region that has short T2 times due to B0 inhomgen.
if add_B0_inhom:
  x0, x1 = np.meshgrid(np.arange(n),np.arange(n), indexing = 'ij')
  w      = gaussian_filter((np.sqrt((x0 - 22*(n//128))**2 + (x1 - 64*(n//128))**2) <= 14).astype(float), 1.5)
  
  T2star_short = (1 - w)*T2star_short + 6*w
  T2star_long  = (1 - w)*T2star_long  + 11*w

#===========================================================================================

# add imag part to
f = np.stack((f,np.zeros(f.shape)), axis = -1)

# create binary mask that shows the inner 64 cube of k-space that is read out
# needed for 0 filling

k0,k1 = np.meshgrid(np.arange(n) - n//2 + 0.5, np.arange(n) - n//2 + 0.5)

# the K-value of 1.5 corresponds to the edge of the cube with radius 32
k_edge = 1.5
abs_k  = k_edge*np.sqrt(k0**2 + k1**2) / 32

abs_k = np.fft.fftshift(abs_k)

#kmask  = np.ones((n,n))
#kmask = gaussian_filter((abs_k <= k_edge).astype(np.float), 1.)
kmask = (abs_k <= k_edge).astype(np.float)

kmask = np.stack((kmask,kmask), axis = -1)

t_read_2d = 1000*readout_time(abs_k)

n_readout_bins = int(32 * (abs_k[kmask[:,:,0] == 1].max() / 1.5))

k_1d = np.linspace(0, abs_k[kmask[:,:,0] == 1].max(), n_readout_bins + 1)

readout_inds = []
t_read_1d    = np.zeros(n_readout_bins)
t_read_2d_binned = np.zeros(t_read_2d.shape)

read_out_img = np.zeros((n,n))

for i in range(n_readout_bins):
  k_start = k_1d[i]
  k_end   = k_1d[i+1]
  rinds   = np.where(np.logical_and(abs_k >= k_start, abs_k <= k_end))

  t_read_1d[i] = t_read_2d[rinds].mean()
  t_read_2d_binned[rinds] = t_read_1d[i]
  readout_inds.append(rinds)
  read_out_img[rinds] = i + 1

apo_imgs  = apo_images(t_read_1d, T2star_short, T2star_long)

#----------------------------------------------------------
#--- simulate the signal

signal = apodized_fft(f, readout_inds, apo_imgs)

# add noise to signal
signal += noise_level*(np.random.randn(*signal.shape))

# multiply signal with readout mas
signal *= kmask

#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
#--- do recons

if T2star_recon_short == -1:
  T2star_short_recon = T2star_short.copy()
else :
  T2star_short_recon = np.zeros((n,n)) + T2star_recon_short

if T2star_recon_long == -1:
  T2star_long_recon = T2star_long.copy()
else :
  T2star_long_recon = np.zeros((n,n)) + T2star_recon_long

apo_imgs_recon = apo_images(t_read_1d, T2star_short_recon, T2star_long_recon)

init_recon  = np.fft.ifft2(np.squeeze(signal.view(dtype = np.complex128))) * np.sqrt(np.prod(f.shape)) / np.sqrt(4*signal.ndim)

init_recon  = init_recon.view('(2,)float')

abs_f           = np.linalg.norm(f,axis=-1)
abs_init_recon  = np.linalg.norm(init_recon,axis=-1)
abs_init_recon *= abs_f.sum() / abs_init_recon.sum()

#----------------------------------------------------------------------------------------
# --- set up stuff for the prior
aimg  = (f.max() - f[...,0])**0.8
aimg += 0.001*aimg.max()*np.random.random(aimg.shape)

# beta = 1e-4 reasonable for inverse crime
s    = np.array([[1,1,1], 
                 [1,0,1], 
                 [1,1,1]])
nnearest = 3 

ninds  = np.zeros((np.prod(aimg.shape),nnearest), dtype = np.uint32)
nearest_neighbors(aimg,s,nnearest,ninds)
ninds2 = is_nearest_neighbor_of(ninds)

#----------------------------------------------------------------------------------------
# (1) recon without regularization

print('LBFGS recon without regularization')
noreg_recon       = init_recon.copy()
noreg_recon_shape = init_recon.shape
noreg_recon       = noreg_recon.flatten()

cost = []
cb = lambda x: cost.append(mr_bowsher_cost(x, noreg_recon_shape, signal, readout_inds, 
                           apo_imgs_recon, 0, ninds, ninds2, method, kmask))

res = fmin_l_bfgs_b(mr_bowsher_cost,
                    noreg_recon, 
                    fprime = mr_bowsher_grad, 
                    args = (noreg_recon_shape, signal, readout_inds, apo_imgs_recon, 0, 
                            ninds, ninds2, method, kmask), 
                    callback = cb,
                    maxiter = niter, 
                    disp = 1)

noreg_recon        = res[0].reshape(noreg_recon_shape)
abs_noreg_recon    = np.linalg.norm(noreg_recon,axis=-1)
abs_noreg_recon_ps = gaussian_filter(abs_noreg_recon, sm_fwhm/2.35)


#----------------------------------------------------------------------------------------
# (2) recon with Bowsher prior
if beta > 0:
  print('LBFGS recon with regularization')
  bow_recon       = init_recon.copy()
  bow_recon_shape = init_recon.shape
  bow_recon       = bow_recon.flatten()
  
  bow_cost = []
  bow_cb   = lambda x: bow_cost.append(mr_bowsher_cost(x, bow_recon_shape, signal, readout_inds, 
                                      apo_imgs_recon, beta, ninds, ninds2, method, kmask))
  
  res = fmin_l_bfgs_b(mr_bowsher_cost,
                      bow_recon, 
                      fprime = mr_bowsher_grad, 
                      args = (bow_recon_shape, signal, readout_inds, apo_imgs_recon, beta, 
                              ninds, ninds2, method, kmask), 
                      callback = bow_cb,
                      maxiter = niter, 
                      disp = 1)
  
  bow_recon     = res[0].reshape(bow_recon_shape)
  abs_bow_recon = np.linalg.norm(bow_recon,axis=-1)

#-----------------------------------------------------------------------------------------
# save the recons
if save_recons:
  output_file = os.path.join('data','recons', '__'.join([x[0] + '_' + str(x[1]) for x in args.__dict__.items()]) + '.h5')
  with h5py.File(output_file, 'w') as hf:
    grp = hf.create_group('images')
    grp.create_dataset('ifft_recon',   data = init_recon)
    grp.create_dataset('noreg_recon',  data = noreg_recon)
    grp.create_dataset('ground_truth', data = f)
    grp.create_dataset('T2star_long',  data = T2star_long)
    grp.create_dataset('T2star_short', data = T2star_short)
    grp.create_dataset('T2star_long_recon',  data = T2star_long_recon)
    grp.create_dataset('T2star_short_recon', data = T2star_short_recon)
    if beta > 0:
      grp.create_dataset('bow_recon',    data = bow_recon)
      grp.create_dataset('prior_image',  data = aimg)

#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------

#--- plot the results

vmax = 1.2*f.max()

fig1, ax1 = py.subplots(3,4, figsize = (12,9), squeeze = False)
ax1[0,0].imshow(abs_f,           vmin = 0,  vmax = vmax)
ax1[0,0].set_title('ground truth')
ax1[0,1].imshow(abs_init_recon,  vmin = 0,  vmax = vmax)
ax1[0,1].set_title('|inverse FFT data|')
ax1[0,2].imshow(abs_noreg_recon, vmin = 0,  vmax = vmax)
ax1[0,2].set_title('|it. recon no prior|')
if beta == 0:
  ax1[0,3].imshow(abs_noreg_recon_ps, vmin = 0,  vmax = vmax)
  ax1[0,3].set_title('ps |it. recon no prior|')
else:
  ax1[0,3].imshow(abs_bow_recon, vmin = 0,  vmax = vmax)
  ax1[0,3].set_title(f'|it. recon bow beta {beta}|')

ax1[1,0].imshow(aimg, vmin = 0, vmax = aimg.max())
ax1[1,0].set_title('anat. prior image')
ax1[1,1].imshow(abs_init_recon - abs_f,     vmin = -0.2*vmax, vmax = 0.2*vmax, cmap = py.cm.bwr)
ax1[1,1].set_title('bias |inverse FFT data|')
ax1[1,2].imshow(abs_noreg_recon - abs_f,    vmin = -0.2*vmax, vmax = 0.2*vmax, cmap = py.cm.bwr)
ax1[1,2].set_title('bias |it. recon no prior|')
if beta == 0:
  ax1[1,3].imshow(abs_noreg_recon_ps - abs_f, vmin = -0.2*vmax, 
                  vmax = 0.2*vmax, cmap = py.cm.bwr)
  ax1[1,3].set_title('bias ps |it. recon no prior|')
else:  
  ax1[1,3].imshow(abs_bow_recon - abs_f, vmin = -0.2*vmax, vmax = 0.2*vmax, cmap = py.cm.bwr)
  ax1[1,3].set_title(f'bias |it. recon bow beta {beta}|')

ax1[2,0].imshow(T2star_long,        vmin = T2star_gm_short, vmax = 1.5*T2star_gm_long)
ax1[2,0].set_title('data T2* long')
ax1[2,1].imshow(T2star_short,       vmin = T2star_gm_short, vmax = 1.5*T2star_gm_short)
ax1[2,1].set_title('data T2* short')
ax1[2,2].imshow(T2star_long_recon,  vmin = T2star_gm_short, vmax = 1.5*T2star_gm_long)
ax1[2,2].set_title('recon T2* long')
ax1[2,3].imshow(T2star_short_recon, vmin = T2star_gm_short, vmax = 1.5*T2star_gm_short)
ax1[2,3].set_title('recon T2* short')

for axx in ax1.flatten(): axx.set_axis_off()
fig1.suptitle(', '.join([x[0] + ':' + str(x[1]) for x in args.__dict__.items()]), fontsize = 'x-small')
fig1.tight_layout(pad = 3)
fig1.savefig(os.path.join('figs', '__'.join([x[0] + '_' + str(x[1]) for x in args.__dict__.items()]) + '_f1.png'))
fig1.show()

# plot the decay envelope
t = t_read_2d.flatten()
k = abs_k.flatten()

fig2, ax2 = py.subplots(3,3, figsize = (12,9), squeeze = False)
ax2[0,0].plot(k, 0.6*np.exp(-t/T2star_gm_short) + 0.4*np.exp(-t/T2star_gm_long), '.', label = 'data')
ax2[0,0].set_title('GM')
ax2[0,1].plot(k, 0.6*np.exp(-t/T2star_wm_short) + 0.4*np.exp(-t/T2star_wm_long), '.')
ax2[0,1].set_title('WM')
ax2[0,2].plot(k, 0.6*np.exp(-t/T2star_csf_short)+ 0.4*np.exp(-t/T2star_csf_long), '.')
ax2[0,2].set_title('CSF')

if (T2star_recon_short != -1) and (T2star_recon_long != -1):
  ax2[0,0].plot(k, 0.6*np.exp(-t/T2star_recon_short) + 0.4*np.exp(-t/T2star_recon_long), '.', label = 'recon')
  ax2[0,1].plot(k, 0.6*np.exp(-t/T2star_recon_short) + 0.4*np.exp(-t/T2star_recon_long), '.')
  ax2[0,2].plot(k, 0.6*np.exp(-t/T2star_recon_short) + 0.4*np.exp(-t/T2star_recon_long), '.')
  
for axx in ax2[0,:]: 
  axx.set_xlabel('|k|')
  axx.set_xlim(0,k_edge)
  axx.set_ylim(0,1)
ax2[0,0].set_ylabel('decay env')

ax2[1,0].plot(t, 0.6*np.exp(-t/T2star_gm_short) + 0.4*np.exp(-t/T2star_gm_long), '.')
ax2[1,0].set_title('GM')
ax2[1,1].plot(t, 0.6*np.exp(-t/T2star_wm_short) + 0.4*np.exp(-t/T2star_wm_long), '.')
ax2[1,1].set_title('WM')
ax2[1,2].plot(t, 0.6*np.exp(-t/T2star_csf_short)+ 0.4*np.exp(-t/T2star_csf_long), '.')
ax2[1,2].set_title('CSF')

if (T2star_recon_short != -1) and (T2star_recon_long != -1):
  ax2[1,0].plot(t, 0.6*np.exp(-t/T2star_recon_short) + 0.4*np.exp(-t/T2star_recon_long), '.')
  ax2[1,1].plot(t, 0.6*np.exp(-t/T2star_recon_short) + 0.4*np.exp(-t/T2star_recon_long), '.')
  ax2[1,2].plot(t, 0.6*np.exp(-t/T2star_recon_short) + 0.4*np.exp(-t/T2star_recon_long), '.')

for axx in ax2[1,:]: 
  axx.set_xlabel('t')
  axx.set_xlim(0,t_read_1d.max())
  axx.set_ylim(0,1)
ax2[1,0].set_ylabel('decay env')

ax2[2,0].plot(t, k, '.')
ax2[2,0].set_xlabel('t')
ax2[2,0].set_ylabel('|k|')

ax2[2,1].semilogy(np.arange(len(cost)) + 1, cost, '.')
if beta > 0:
  ax2[2,1].semilogy(np.arange(len(bow_cost)) + 1, bow_cost, '.')
ax2[2,1].set_xlabel('iteration')
ax2[2,1].set_ylabel('cost')

ax2[2,2].plot(abs_f[n//2,60:110], 'k', label = 'gt')
ax2[2,2].plot(abs_init_recon[n//2,60:110], 'r--', label = 'ifft')
if noise_level == 0:
  ax2[2,2].plot(abs_noreg_recon[n//2,60:110], 'g--', label = 'it. no p.')
else:
  ax2[2,2].plot(abs_noreg_recon_ps[n//2,60:110], 'g--', label = 'ps it. no p.')
if beta > 0:
  ax2[2,2].plot(abs_bow_recon[n//2,60:110], 'b--', label = 'it. bow')
ax2[2,2].set_ylim(0.6,1.4)
ax2[2,2].set_title('line profile')

ax2[0,0].legend()
ax2[2,2].legend()

for axx in ax2.flatten(): 
  axx.grid(ls = ':')

fig2.suptitle(', '.join([x[0] + ':' + str(x[1]) for x in args.__dict__.items()]), fontsize = 'x-small')
fig2.tight_layout(pad = 3)
fig2.savefig(os.path.join('figs', '__'.join([x[0] + '_' + str(x[1]) for x in args.__dict__.items()]) + '_f2.png'))
fig2.show()

import os
import numpy as np
import cupy as cp
import h5py
import nibabel as nib

import matplotlib as mpl
import matplotlib.pyplot as py

from scipy.optimize import fmin_l_bfgs_b, fmin_cg
from nearest_neighbors import nearest_neighbors, is_nearest_neighbor_of
from readout_time import readout_time
from apodized_fft import apodized_fft_multi_echo
from cost_functions import multi_echo_bowsher_cost, multi_echo_bowsher_grad, multi_echo_bowsher_cost_gamma, multi_echo_bowsher_grad_gamma

from scipy.ndimage     import zoom, gaussian_filter
from scipy.interpolate import interp1d

from argparse import ArgumentParser
from datetime import datetime

import pymirc.viewer as pv
#--------------------------------------------------------------
#--------------------------------------------------------------

parser = ArgumentParser(description = '3D na mr dual echo simulation')
parser.add_argument('--niter',  default = 10, type = int)
parser.add_argument('--n_outer', default = 3, type = int)
parser.add_argument('--bet_recon', default = 1., type = float)
parser.add_argument('--bet_gam', default = 1., type = float)
parser.add_argument('--nnearest', default = 4,  type = int)
parser.add_argument('--nneigh',   default = 20,  type = int, choices = [8,20])
parser.add_argument('--n',   default = 128,  type = int, choices = [128,256])
parser.add_argument('--slice2d', default = 56, type = int)

args = parser.parse_args()

niter       = args.niter
n_outer     = args.n_outer
bet_recon   = args.bet_recon
bet_gam     = args.bet_gam
nnearest    = args.nnearest 
nneigh      = args.nneigh
n           = args.n
slice2d     = args.slice2d

method      = 0
delta_t     = 5.

#--------------------------------------------------------------
#--------------------------------------------------------------


#-------------------
# load the data
#-------------------

pdir   = os.path.join('data','sodium_data','TBI-n005','PhyCha_kw0_preprocessed')
odir   = os.path.join(pdir, datetime.now().strftime("%y%m%d-%H%M%S") + '_2D_' +'__'.join([x[0] + '_' + str(x[1]) for x in args.__dict__.items()]))

if not os.path.exists(odir):
  os.makedirs(odir)

# write input arguments to file
with open(os.path.join(odir,'input_params.csv'), 'w') as f:
  for x in args.__dict__.items():
    f.write("%s,%s\n"%(x[0],x[1]))

nechos = 2

t1_vol = np.load(os.path.join(pdir,f't1_coreg_{n}.npy'))
sens   = np.load(os.path.join(pdir,f'sens_{n}.npy')).astype(np.complex128).view('(2,)float')
echo1  = np.load(os.path.join(pdir,f'echo1_{n}.npy')).astype(np.complex128).view('(2,)float')
echo2  = np.load(os.path.join(pdir,f'echo2_{n}.npy')).astype(np.complex128).view('(2,)float')

ncoils = echo1.shape[0]

signal = np.zeros((ncoils,nechos) + echo1.shape[1:])

for i in range(ncoils):
  signal[i,0,...] = echo1[i,...]
  signal[i,1,...] = echo2[i,...]

#-------------------
# extract 2D signal
#-------------------

k0,k1,k2 = np.meshgrid(np.arange(n) - n//2 + 0.5, np.arange(n) - n//2 + 0.5, np.arange(n) - n//2 + 0.5)

signal3d = signal.copy()
signal   = np.zeros(signal3d.shape[:-2] + (2,))

# create the han window that we need to multiply to the mask
h_win_z = interp1d(np.arange(32), np.hanning(64)[32:], fill_value = 0, bounds_error = False)
# abs_k was scaled to have the k edge at 32, we have to revert that for the han window
hmask_z = h_win_z(np.abs(np.fft.fftshift(k2).flatten())).reshape(n,n,n)

for j in range(ncoils):
  for i in range(nechos):
    s = signal3d[j,i,...].view(dtype = np.complex128).squeeze().copy()
    signal[j,i,...] = np.ascontiguousarray(np.fft.ifftn(s*hmask_z, norm = 'ortho', axes = (-1,))[...,slice2d].view('(2,)float'))

sens3d = sens.copy()
sens   = sens3d[...,slice2d,:]

#----------------------
# calc 2D readout times
#----------------------

k0,k1 = np.meshgrid(np.arange(n) - n//2 + 0.5, np.arange(n) - n//2 + 0.5)
abs_k = np.sqrt(k0**2 + k1**2)
abs_k = np.fft.fftshift(abs_k)

# rescale abs_k such that k = 1.5 is at r = 32 (the edge)
k_edge = 1.5
abs_k *= k_edge/32

# recalculate 2D readout times and indices
t_read_2d = 1000*readout_time(abs_k)

n_readout_bins = 32

k_1d = np.linspace(0, k_edge, n_readout_bins + 1)

readout_inds = []
tr= np.zeros(n_readout_bins)
t_read_2d_binned = np.zeros(t_read_2d.shape)

read_out_img = np.zeros((n,n))

for i in range(n_readout_bins):
  k_start = k_1d[i]
  k_end   = k_1d[i+1]
  rinds   = np.where(np.logical_and(abs_k >= k_start, abs_k <= k_end))

  tr[i] = t_read_2d[rinds].mean()
  t_read_2d_binned[rinds] = tr[i]
  readout_inds.append(rinds)
  read_out_img[rinds] = i + 1

#------------
#------------

kmask  = np.zeros(signal.shape)
for j in range(ncoils):
  for i in range(nechos):
    kmask[j,i,...,0] = (read_out_img > 0).astype(np.float)
    kmask[j,i,...,1] = (read_out_img > 0).astype(np.float)

# multiply signal with readout mask
signal *= kmask
abs_signal = np.linalg.norm(signal, axis = -1)

ifft          = np.zeros(signal.shape)
ifft_filtered = np.zeros(signal.shape)
abs_ifft          = np.zeros(signal.shape[:-1])
abs_ifft_filtered = np.zeros(signal.shape[:-1])

# create the han window that we need to multiply to the mask
h_win = interp1d(np.arange(32), np.hanning(64)[32:], fill_value = 0, bounds_error = False)
# abs_k was scaled to have the k edge at 32, we have to revert that for the han window
hmask = h_win(abs_k.flatten()*32/k_edge).reshape(n,n)

for j in range(ncoils):
  for i in range(nechos):
    s = signal[j,i,...].view(dtype = np.complex128).squeeze().copy()
    ifft[j,i,...] = np.ascontiguousarray(np.fft.ifftn(s, norm = 'ortho').view('(2,)float'))
    ifft_filtered[j,i,...] = np.ascontiguousarray(np.fft.ifftn(hmask*s, norm = 'ortho').view('(2,)float'))
    abs_ifft[j,i,...] = np.linalg.norm(ifft[j,i,...], axis = -1)
    abs_ifft_filtered[j,i,...] = np.linalg.norm(ifft_filtered[j,i,...], axis = -1)

#----------------------------------------------------------------------------------------
# --- set up stuff for the prior
aimg = t1_vol.copy()[...,slice2d]

if nneigh == 8:
  s    = np.array([[1,1,1], 
                   [1,0,1], 
                   [1,1,1]])
                   
elif nneigh == 20:
  s = np.array([[0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0]])
 
  
ninds  = np.zeros((np.prod(aimg.shape),nnearest), dtype = np.uint32)
nearest_neighbors(aimg,s,nnearest,ninds)
ninds2 = is_nearest_neighbor_of(ninds)

#--------------------------------------------------------------------------------------------------
# initialize variables

tmp1 = abs_ifft_filtered[:,0,...].sum(0)
tmp2 = abs_ifft_filtered[:,1,...].sum(0)

Gam_recon = tmp2 / tmp1
Gam_recon[tmp1 < 0.05*tmp1.max()] = 1
Gam_recon[Gam_recon > 1] = 1

recon = np.zeros(signal.shape[2:])
recon[...,0] = tmp1

recon_shape = recon.shape
Gam_bounds  = (n**2)*[(0.001,1)]

# rescale signal and initial recon to get an image with max approx 1
scale_fac = 1./tmp1.max()

signal *= scale_fac
tmp1   *= scale_fac
tmp2   *= scale_fac
recon  *= scale_fac

abs_recon   = np.linalg.norm(recon,axis=-1)


cost1 = []
cost2 = []

fig1, ax1 = py.subplots(2,n_outer+1, figsize = ((n_outer+1)*3,6))
vmax = 1.2
ax1[0,0].imshow(Gam_recon.T, vmin = 0, vmax = 1, cmap = py.cm.Greys_r, origin = 'lower')
ax1[1,0].imshow(abs_recon.T, vmin = 0, vmax = vmax, cmap = py.cm.Greys_r, origin = 'lower')

#--------------------------------------------------------------------------------------------------

for i in range(n_outer):

  print('LBFGS to optimize for recon')
  
  recon       = recon.flatten()
  
  cb = lambda x: cost2.append(multi_echo_bowsher_cost(x, recon_shape, signal, readout_inds, 
                             Gam_recon, tr, delta_t, nechos, kmask, bet_recon, ninds, ninds2, method, sens))
  
  res = fmin_l_bfgs_b(multi_echo_bowsher_cost,
                      recon, 
                      fprime = multi_echo_bowsher_grad, 
                      args = (recon_shape, signal, readout_inds, 
                              Gam_recon, tr, delta_t, nechos, kmask, bet_recon, ninds, ninds2, method, sens),
                      callback = cb,
                      maxiter = niter, 
                      disp = 1)
  
  recon        = res[0].reshape(recon_shape)
  abs_recon    = np.linalg.norm(recon,axis=-1)
  
  ax1[1,i+1].imshow(abs_recon.T, vmin = 0, vmax = vmax, cmap = py.cm.Greys_r, origin = 'lower')

  #---------------------------------------

  print('LBFGS to optimize for gamma')
  
  Gam_recon = Gam_recon.flatten()
  
  cb = lambda x: cost1.append(multi_echo_bowsher_cost_gamma(x, recon_shape, signal, readout_inds, 
                             recon, tr, delta_t, nechos, kmask, bet_gam, ninds, ninds2, method, sens))
  
  res = fmin_l_bfgs_b(multi_echo_bowsher_cost_gamma,
                      Gam_recon, 
                      fprime = multi_echo_bowsher_grad_gamma, 
                      args = (recon_shape, signal, readout_inds, 
                              recon, tr, delta_t, nechos, kmask, bet_gam, ninds, ninds2, method, sens),
                      callback = cb,
                      maxiter = niter, 
                      bounds = Gam_bounds,
                      disp = 1)
  
  Gam_recon = res[0].reshape(recon_shape[:-1])

  # reset values in low signal regions
  Gam_recon[tmp1 < 0.05*tmp1.max()] = 1

  ax1[0,i+1].imshow(Gam_recon.T, vmin = 0, vmax = 1, cmap = py.cm.Greys_r, origin = 'lower')

#--------------------------------------------------------------------------------------------------

fig1.tight_layout()
fig1.show()
fig1.savefig(os.path.join(odir,'convergence.png'))

# generate the sum of squares image
ref_recon = abs_ifft[:,0,...].sum(0)*scale_fac
# scale total of ref_recon to joint recon
ref_recon *= (np.percentile(abs_recon,99.99) / np.percentile(ref_recon,99.99))

ref_recon_filt = abs_ifft_filtered[:,0,...].sum(0)*scale_fac
# scale total of ref_recon to joint recon
ref_recon_filt *= (np.percentile(abs_recon,99.99) / np.percentile(ref_recon_filt,99.99))

# save the recons
output_file = os.path.join(odir, 'recons.h5')
with h5py.File(output_file, 'w') as hf:
  grp = hf.create_group('images')
  grp.create_dataset('Gam_recon',     data = Gam_recon)
  grp.create_dataset('recon',         data = recon)
  grp.create_dataset('abs_recon',     data = abs_recon)
  grp.create_dataset('ifft',          data = ref_recon)
  grp.create_dataset('ifft_filt',     data = ref_recon_filt)
  grp.create_dataset('prior_image',   data = aimg)

#--------------------------------------------------------------------------------------------------

fig2, ax2 = py.subplots(2,3, figsize = (9,6))
ax2[0,0].imshow(aimg.T, cmap = py.cm.Greys_r, vmin = 0, origin = 'lower')
ax2[0,0].set_title('prior image')
ax2[0,1].imshow(ref_recon.T, cmap = py.cm.Greys_r, vmin = 0, vmax = vmax, origin = 'lower')
ax2[0,1].set_title('SOS IFFT')
ax2[0,2].imshow(ref_recon_filt.T, cmap = py.cm.Greys_r, vmin = 0, vmax = vmax, origin = 'lower')
ax2[0,2].set_title('SOS IFFT filtered')
ax2[1,1].imshow(abs_recon.T, cmap = py.cm.Greys_r, vmin = 0, vmax = vmax, origin = 'lower')
ax2[1,1].set_title('it. joint Na recon')
ax2[1,2].imshow(Gam_recon.T, cmap = py.cm.Greys_r, vmin = 0, vmax = 1, origin = 'lower')
ax2[1,2].set_title('it. joint Gamma recon')

for axx in ax2.flatten(): axx.set_axis_off()

fig2.tight_layout()
fig2.savefig(os.path.join(odir,'results.png'))
fig2.show()

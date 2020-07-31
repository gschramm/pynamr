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

#--------------------------------------------------------------
#--------------------------------------------------------------

parser = ArgumentParser(description = '3D na mr dual echo simulation')
parser.add_argument('--niter',  default = 10, type = int)
parser.add_argument('--niter_last', default = 10, type = int)
parser.add_argument('--n_outer', default = 3, type = int)
parser.add_argument('--method', default = 0, type = int)
parser.add_argument('--bet_recon', default = 5., type = float)
parser.add_argument('--bet_gam', default = 5., type = float)
parser.add_argument('--delta_t', default = 5., type = float)
parser.add_argument('--nnearest', default = 13,  type = int)
parser.add_argument('--nneigh',   default = 80,  type = int, choices = [18,80])

args = parser.parse_args()

niter       = args.niter
niter_last  = args.niter_last
n_outer     = args.n_outer
bet_recon   = args.bet_recon
bet_gam     = args.bet_gam
method      = args.method
delta_t     = args.delta_t
nnearest    = args.nnearest 
nneigh      = args.nneigh

n = 128

#odir = os.path.join('data','recons_multi_3d', '__'.join([x[0] + '_' + str(x[1]) for x in args.__dict__.items()]))
#
#if not os.path.exists(odir):
#    os.makedirs(odir)
#
## write input arguments to file
#with open(os.path.join(odir,'input_params.csv'), 'w') as f:
#  for x in args.__dict__.items():
#    f.write("%s,%s\n"%(x[0],x[1]))

#--------------------------------------------------------------
#--------------------------------------------------------------

#-------------------
# load the data
#-------------------

pdir = 'W:\georg\sodium_bowsher\BT-007_visit2'

na_nii   = nib.load(os.path.join(pdir,'TE03_128_PhyCha_kw1.nii'))
na2_nii  = nib.load(os.path.join(pdir,'TE5_128_PhyCha_kw1.nii'))
t1_nii   = nib.load(os.path.join(pdir,'mprage_128_aligned.nii'))

na_nii  = nib.as_closest_canonical(na_nii)
na2_nii = nib.as_closest_canonical(na2_nii)
t1_nii  = nib.as_closest_canonical(t1_nii)

na_vol  = na_nii.get_fdata()
na2_vol = na2_nii.get_fdata()
t1_vol  = t1_nii.get_fdata()

f = na_vol.astype(np.float64)
f = np.stack((f,np.zeros(f.shape)), axis = -1)

f2 = na2_vol.astype(np.float64)
f2 = np.stack((f2,np.zeros(f2.shape)), axis = -1)

signal = np.array([np.fft.fftn(f[...,0], norm = 'ortho').view('(2,)float'), 
                   np.fft.fftn(f2[...,0], norm = 'ortho').view('(2,)float')])

#-------------------
# calc readout times
#-------------------

# setup the frequency array as used in numpy fft
k0,k1,k2 = np.meshgrid(np.arange(n) - n//2 + 0.5, np.arange(n) - n//2 + 0.5, np.arange(n) - n//2 + 0.5)
abs_k = np.sqrt(k0**2 + k1**2 + k2**2)
abs_k = np.fft.fftshift(abs_k)

# rescale abs_k such that k = 1.5 is at r = 32 (the edge)
k_edge = 1.5
abs_k *= k_edge/32

# calculate the readout times and the k-spaces locations that
# are read at a given time
t_read_3 = 1000*readout_time(abs_k)

n_readout_bins = 32

k_1d = np.linspace(0, k_edge, n_readout_bins + 1)

readout_inds = []
tr= np.zeros(n_readout_bins)
t_read_3_binned = np.zeros(t_read_3.shape)

read_out_img = np.zeros((n,n,n))

for i in range(n_readout_bins):
  k_start = k_1d[i]
  k_end   = k_1d[i+1]
  rinds   = np.where(np.logical_and(abs_k >= k_start, abs_k <= k_end))

  tr[i] = t_read_3[rinds].mean()
  t_read_3_binned[rinds] = tr[i]
  readout_inds.append(rinds)
  read_out_img[rinds] = i + 1

#------------
#------------

kmask  = np.ones(signal.shape)

ifft          = np.zeros((2,) + signal.shape)
ifft_filtered = np.zeros((2,) + signal.shape)
abs_ifft          = np.zeros((2,) + signal.shape[:-1])
abs_ifft_filtered = np.zeros((2,) + signal.shape[:-1])

#----------------------------------------------------------------------------------------
# --- set up stuff for the prior
aimg = t1_vol.copy() / t1_vol.max()

if nneigh == 18:
  s    = np.array([[[0,1,0], 
                    [1,1,1], 
                    [0,1,0]],
                   [[1,1,1], 
                    [1,0,1], 
                    [1,1,1]],
                   [[0,1,0], 
                    [1,1,1], 
                    [0,1,0]]])
elif nneigh == 80:
  s = np.array([[[0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 0],
                 [0, 1, 1, 1, 0],
                 [0, 1, 1, 1, 0],
                 [0, 0, 0, 0, 0]],
                [[0, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 0]],
                [[0, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1],
                 [1, 1, 0, 1, 1],
                 [1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 0]],
                [[0, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 0]],
                [[0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 0],
                 [0, 1, 1, 1, 0],
                 [0, 1, 1, 1, 0],
                 [0, 0, 0, 0, 0]]])

  
ninds  = np.zeros((np.prod(aimg.shape),nnearest), dtype = np.uint32)
nearest_neighbors(aimg,s,nnearest,ninds)
ninds2 = is_nearest_neighbor_of(ninds)

#--------------------------------------------------------------------------------------------------
# initialize variables

Gam_recon = f2[...,0] / f[...,0]
Gam_recon[f[...,0] < 0.1*f[...,0].max()] = 1
Gam_recon[Gam_recon > 1] = 1

recon = f.copy()
recon_shape = recon.shape
abs_recon   = np.linalg.norm(recon,axis=-1)

Gam_bounds = (n**3)*[(0.001,1)]

cost1 = []
cost2 = []

#fig1, ax1 = py.subplots(4,n_outer+1, figsize = ((n_outer+1)*3,12))
#vmax = 1.5*abs_f.max()
#ax1[0,0].imshow(Gam_recon[...,64], vmin = 0, vmax = 1, cmap = py.cm.Greys_r)
#ax1[1,0].imshow(Gam_recon[...,64] - Gam[...,64], vmin = -0.3, vmax = 0.3, cmap = py.cm.bwr)
#ax1[2,0].imshow(abs_recon[...,64], vmin = 0, vmax = vmax, cmap = py.cm.Greys_r)
#ax1[3,0].imshow(abs_recon[...,64] - abs_f[...,64], vmax = 0.3, vmin = -0.3, cmap = py.cm.bwr)
#
#--------------------------------------------------------------------------------------------------

for i in range(n_outer):

  print('LBFGS to optimize for recon')
  
  if i == (n_outer - 1):
    niter_recon = niter_last
  else:
    niter_recon = niter

  recon       = recon.flatten()
  
  cb = lambda x: cost2.append(multi_echo_bowsher_cost(x, recon_shape, signal, readout_inds, 
                             Gam_recon, tr, delta_t, 2, kmask, bet_recon, ninds, ninds2, method))
  
  res = fmin_l_bfgs_b(multi_echo_bowsher_cost,
                      recon, 
                      fprime = multi_echo_bowsher_grad, 
                      args = (recon_shape, signal, readout_inds, 
                              Gam_recon, tr, delta_t, 2, kmask, bet_recon, ninds, ninds2, method),
                      callback = cb,
                      maxiter = niter_recon, 
                      disp = 1)
  
  recon        = res[0].reshape(recon_shape)
  abs_recon    = np.linalg.norm(recon,axis=-1)
  
  #ax1[2,i+1].imshow(abs_recon[...,64], vmin = 0, vmax = vmax, cmap = py.cm.Greys_r)
  #ax1[3,i+1].imshow(abs_recon[...,64] - abs_f[...,64], vmax = 0.3, vmin = -0.3, cmap = py.cm.bwr)

  #---------------------------------------

  print('LBFGS to optimize for gamma')
  
  Gam_recon = Gam_recon.flatten()
  
  cb = lambda x: cost1.append(multi_echo_bowsher_cost_gamma(x, recon_shape, signal, readout_inds, 
                             recon, tr, delta_t, 2, kmask, bet_gam, ninds, ninds2, method))
  
  res = fmin_l_bfgs_b(multi_echo_bowsher_cost_gamma,
                      Gam_recon, 
                      fprime = multi_echo_bowsher_grad_gamma, 
                      args = (recon_shape, signal, readout_inds, 
                              recon, tr, delta_t, 2, kmask, bet_gam, ninds, ninds2, method),
                      callback = cb,
                      maxiter = niter, 
                      bounds = Gam_bounds,
                      disp = 1)
  
  Gam_recon = res[0].reshape(recon_shape[:-1])

  # reset values in low signal regions
  Gam_recon[f[...,0] < 0.1*f[...,0].max()] = 1

  #ax1[0,i+1].imshow(Gam_recon[...,64], vmin = 0, vmax = 1, cmap = py.cm.Greys_r)
  #ax1[1,i+1].imshow(Gam_recon[...,64] - Gam[...,64], vmin = -0.3, vmax = 0.3, cmap = py.cm.bwr)

import pymirc.viewer as pv
ims = 4*[{'cmap':py.cm.Greys_r}]
pv.ThreeAxisViewer([np.flip(f[...,0],1),np.flip(abs_recon,1), np.flip(Gam_recon,1), np.flip(aimg,1)], imshow_kwargs = ims)

##--------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------
#
#fig1.tight_layout()
#fig1.savefig(os.path.join(odir,'convergence.png'))
#fig1.show()
#
## save the recons
#output_file = os.path.join(odir, 'recons.h5')
#with h5py.File(output_file, 'w') as hf:
#  grp = hf.create_group('images')
#  grp.create_dataset('Gam',           data = Gam)
#  grp.create_dataset('Gam_recon',     data = Gam_recon)
#  grp.create_dataset('ground_truth',  data = f)
#  grp.create_dataset('signal',        data = signal)
#  grp.create_dataset('recon',         data = recon)
#  grp.create_dataset('ifft',          data = ifft)
#  grp.create_dataset('ifft_filtered', data = ifft_filtered)
#  grp.create_dataset('prior_image',   data = aimg)
#
##--------------------------------------------------------------------------------------------------
#
#
#ims1 = 2*[{'cmap':py.cm.Greys_r}] + [{'cmap':py.cm.Greys_r, 'vmin':0, 'vmax':vmax}]
#vi1 = pv.ThreeAxisViewer([abs_ifft[0,...],abs_ifft_filtered[0,...],abs_f], imshow_kwargs = ims1)
#vi1.fig.savefig(os.path.join(odir,'fig1.png'))
#
#ims2 = [{'cmap':py.cm.Greys_r, 'vmin':0, 'vmax':vmax}] + 2*[{'cmap':py.cm.Greys_r, 'vmin':0, 'vmax':1.}]
#vi2  = pv.ThreeAxisViewer([abs_recon,Gam_recon, Gam], imshow_kwargs = ims2)
#vi2.fig.savefig(os.path.join(odir,'fig2.png'))

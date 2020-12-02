import os
import numpy as np
import cupy as cp
import h5py

import matplotlib as mpl
import matplotlib.pyplot as py

from scipy.optimize import fmin_l_bfgs_b, fmin_cg
from nearest_neighbors import nearest_neighbors, is_nearest_neighbor_of
from readout_time import readout_time
from apodized_fft import apodized_fft_multi_echo
from cost_functions import multi_echo_bowsher_cost, multi_echo_bowsher_grad, multi_echo_bowsher_cost_gamma
from cost_functions import multi_echo_bowsher_grad_gamma, multi_echo_bowsher_cost_total

from scipy.ndimage     import gaussian_filter
from scipy.interpolate import interp1d

from pymirc.metrics import neg_mutual_information
from pymirc.image_operations import zoom3d

from argparse import ArgumentParser

import nibabel as nib
import pymirc.viewer as pv
#--------------------------------------------------------------
#--------------------------------------------------------------

parser = ArgumentParser(description = '3D na mr dual echo phantom recon')
parser.add_argument('--niter',  default = 10, type = int)
parser.add_argument('--n_outer', default = 6, type = int)
parser.add_argument('--method', default = 0, type = int)
parser.add_argument('--bet_recon', default = 0.1, type = float)
parser.add_argument('--bet_gam', default = 0.3, type = float)
parser.add_argument('--delta_t', default = 5., type = float)
parser.add_argument('--nnearest', default = 13,  type = int)
parser.add_argument('--nneigh',   default = 80,  type = int, choices = [18,80])
parser.add_argument('--noise_level', default = 0.4,  type = float)

args = parser.parse_args()

niter       = args.niter
n_outer     = args.n_outer
bet_recon   = args.bet_recon
bet_gam     = args.bet_gam
method      = args.method
n           = 128
nechos      = 2
delta_t     = args.delta_t
nnearest    = args.nnearest 
nneigh      = args.nneigh
ncoils      = 1
noise_level = args.noise_level

# scaling factor to get max(recon) close to 1
scale_fac   = 85

odir = os.path.join('data','recons_multi_3d_phantom', '__'.join([x[0] + '_' + str(x[1]) for x in args.__dict__.items()]))

if not os.path.exists(odir):
    os.makedirs(odir)

# write input arguments to file
with open(os.path.join(odir,'input_params.csv'), 'w') as f:
  for x in args.__dict__.items():
    f.write("%s,%s\n"%(x[0],x[1]))

#--------------------------------------------------------------
#--------------------------------------------------------------

np.random.seed(0)
  
#-------------------
# load data
#-------------------

t1_vol = nib.load('data/New_Phantom_MTE/HR_phantom_1h_1mm.nii').get_fdata()

echo1 = np.flip(np.fromfile('data/New_Phantom_MTE/HR_phantom_23Na_4mm_MPSF_0p5_g20_TE0p4.cpx', dtype = np.complex64).reshape((n,n,n)), (0,2))
echo2 = np.flip(np.fromfile('data/New_Phantom_MTE/HR_phantom_23Na_4mm_MPSF_0p5_g20_TE5p1.cpx', dtype = np.complex64).reshape((n,n,n)), (0,2))

signal = np.zeros((ncoils,nechos,n,n,n,2))

# fftshift remains unclear
signal[0,0,...] = np.roll(np.fft.fftn(echo1, norm = 'ortho').view('(2,)float'), (64,64,64), axis = (0,1,2)) 
signal[0,1,...] = np.roll(np.fft.fftn(echo2, norm = 'ortho').view('(2,)float'), (64,64,64), axis = (0,1,2)) 

# scale signal and add noise
signal /= scale_fac

if noise_level > 0:
  signal += noise_level*(np.random.randn(*signal.shape))*np.sqrt(nechos)/np.sqrt(2)

sens = np.zeros((1,n,n,n,2))
sens[...,0] = 1

# align T1 to 128 grid
t1_vol = np.pad(zoom3d(t1_vol,0.45), ((5,7), (7,5), (5,7)))

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
hmask = h_win(abs_k.flatten()*32/k_edge).reshape(n,n,n)

for j in range(ncoils):
  for i in range(nechos):
    s = signal[j,i,...].view(dtype = np.complex128).squeeze().copy()
    ifft[j,i,...] = np.ascontiguousarray(np.fft.ifftn(s, norm = 'ortho').view('(2,)float'))
    ifft_filtered[j,i,...] = np.ascontiguousarray(np.fft.ifftn(hmask*s, norm = 'ortho').view('(2,)float'))
    abs_ifft[j,i,...] = np.linalg.norm(ifft[j,i,...], axis = -1)
    abs_ifft_filtered[j,i,...] = np.linalg.norm(ifft_filtered[j,i,...], axis = -1)

#----------------------------------------------------------------------------------------
# --- set up stuff for the prior
aimg = t1_vol / t1_vol.max()

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

tmp1 = abs_ifft_filtered[:,0,...].sum(0)
tmp2 = abs_ifft_filtered[:,1,...].sum(0)

Gam_recon = tmp2 / tmp1
Gam_recon[tmp1 < 0.05*tmp1.max()] = 1
Gam_recon[Gam_recon > 1] = 1

recon = np.zeros(signal.shape[2:])
recon[...,0] = tmp1

recon_shape = recon.shape
Gam_bounds  = ((Gam_recon.shape[0])**3)*[(0.001,1)]

abs_recon   = np.linalg.norm(recon,axis=-1)

cost = []

vmax = 1.5
fig1, ax1 = py.subplots(2,n_outer+1, figsize = ((n_outer+1)*3,6))
ax1[0,0].imshow(Gam_recon[...,64].T, vmin = 0.5, vmax = 1, cmap = py.cm.Greys_r)
ax1[1,0].imshow(abs_recon[...,64].T, vmin = 0, vmax = vmax, cmap = py.cm.Greys_r)

#--------------------------------------------------------------------------------------------------

for i in range(n_outer):

  print('LBFGS to optimize for recon')
  
  recon       = recon.flatten()
  
  cb = lambda x: cost.append(multi_echo_bowsher_cost_total(x, recon_shape, signal, readout_inds, 
                             Gam_recon, tr, delta_t, nechos, kmask, bet_recon, bet_gam, ninds, method, sens, 0))
  
  res = fmin_l_bfgs_b(multi_echo_bowsher_cost,
                      recon, 
                      fprime = multi_echo_bowsher_grad, 
                      args = (recon_shape, signal, readout_inds, 
                              Gam_recon, tr, delta_t, nechos, kmask, bet_recon, ninds, ninds2, method, sens,0),
                      callback = cb,
                      maxiter = niter, 
                      disp = 1)
  
  recon        = res[0].reshape(recon_shape)
  abs_recon    = np.linalg.norm(recon,axis=-1)
  
  ax1[1,i+1].imshow(abs_recon[...,64].T, vmin = 0, vmax = vmax, cmap = py.cm.Greys_r)

  #---------------------------------------

  print('LBFGS to optimize for gamma')
  
  Gam_recon = Gam_recon.flatten()
  
  cb = lambda x: cost.append(multi_echo_bowsher_cost_total(recon, recon_shape, signal, readout_inds, 
                             x, tr, delta_t, nechos, kmask, bet_recon, bet_gam, ninds, method, sens, 0))
  
  res = fmin_l_bfgs_b(multi_echo_bowsher_cost_gamma,
                      Gam_recon, 
                      fprime = multi_echo_bowsher_grad_gamma, 
                      args = (recon_shape, signal, readout_inds, 
                              recon, tr, delta_t, nechos, kmask, bet_gam, ninds, ninds2, method, sens, 0),
                      callback = cb,
                      maxiter = niter, 
                      bounds = Gam_bounds,
                      disp = 1)
  
  Gam_recon = res[0].reshape(recon_shape[:-1])

  # reset values in low signal regions
  #Gam_recon[tmp1 < 0.05*tmp1.max()] = 1
  Gam_recon[abs_recon < 0.05*abs_recon.max()] = 1

  ax1[0,i+1].imshow(Gam_recon[...,64].T, vmin = 0.5, vmax = 1, cmap = py.cm.Greys_r)

#--------------------------------------------------------------------------------------------------

fig1.tight_layout()
fig1.show()
fig1.savefig(os.path.join(odir,'convergence.png'))

# generate the sum of squares image
ref_recon = abs_ifft[:,0,...].sum(0)
# scale total of ref_recon to joint recon
ref_recon *= (np.percentile(abs_recon,99.99) / np.percentile(ref_recon,99.99))

ref_recon_filt = abs_ifft_filtered[:,0,...].sum(0)
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
  grp.create_dataset('cost',          data = cost)

#--------------------------------------------------------------------------------------------------

sl = 64

fig2,ax2 = py.subplots(2,3, figsize = (3*3.2,2*3))
im00 = ax2[0,0].imshow(ref_recon[...,sl].T, vmin = 0, vmax = vmax, cmap = py.cm.Greys_r) 
ax2[0,0].set_title('IFFT')
im01 = ax2[0,1].imshow(ref_recon_filt[...,sl].T, vmin = 0, vmax = vmax, cmap = py.cm.Greys_r) 
ax2[0,1].set_title('filt. IFFT')
im02 = ax2[0,2].imshow(aimg[...,sl].T, cmap = py.cm.Greys_r) 
ax2[0,2].set_title('anat. prior. img.')
im10 = ax2[1,0].imshow(abs_recon[...,sl].T, vmin = 0, vmax = vmax, cmap = py.cm.Greys_r) 
ax2[1,0].set_title(f'joint Na b={bet_recon}')
im11 = ax2[1,1].imshow(Gam_recon[...,sl].T, vmin = 0.5, vmax = 1, cmap = py.cm.Greys_r) 
ax2[1,1].set_title(f'joint Gam b={bet_gam}')

for axx in ax2.flatten():
  axx.set_xticks([])
  axx.set_yticks([])

im12 = ax2[1,2].set_axis_off()

fig2.colorbar(im00, ax = ax2[0,0])
fig2.colorbar(im01, ax = ax2[0,1])
fig2.colorbar(im02, ax = ax2[0,2])
fig2.colorbar(im10, ax = ax2[1,0])
fig2.colorbar(im11, ax = ax2[1,1])

fig2.tight_layout()
fig2.show()
fig1.savefig(os.path.join(odir,'results_transverse.png'))


ims2 = [{'cmap':py.cm.Greys_r}] + 3*[{'cmap':py.cm.Greys_r, 'vmin':0, 'vmax':vmax}] + [{'cmap':py.cm.Greys_r, 'vmin':0.5, 'vmax':1.}]
vi2  = pv.ThreeAxisViewer([aimg, ref_recon,ref_recon_filt,abs_recon,Gam_recon], imshow_kwargs = ims2)
vi2.fig.savefig(os.path.join(odir,'fig1.png'))

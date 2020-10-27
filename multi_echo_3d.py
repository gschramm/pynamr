import os
import numpy as np
import cupy as cp
import h5py
import math

import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as py

from scipy.optimize import fmin_l_bfgs_b, fmin_cg
from nearest_neighbors import nearest_neighbors, is_nearest_neighbor_of
from readout_time import readout_time
from apodized_fft import apodized_fft_multi_echo
from cost_functions import multi_echo_bowsher_cost, multi_echo_bowsher_grad, multi_echo_bowsher_cost_gamma
from cost_functions import multi_echo_bowsher_grad_gamma, multi_echo_bowsher_cost_total

from scipy.ndimage     import zoom, gaussian_filter
from scipy.interpolate import interp1d

import pymirc.viewer as pv
from pymirc.image_operations import aff_transform, zoom3d

from argparse import ArgumentParser

#--------------------------------------------------------------
#--------------------------------------------------------------

parser = ArgumentParser(description = '3D na mr dual echo simulation')
parser.add_argument('--niter',  default = 20, type = int)
parser.add_argument('--n_outer', default = 12, type = int)
parser.add_argument('--method', default = 0, type = int)
parser.add_argument('--bet_recon', default = 0.01, type = float)
parser.add_argument('--bet_gam', default = 0.03, type = float)
parser.add_argument('--n', default = 128, type = int, choices = [128,256])
parser.add_argument('--noise_level', default = 0.4,  type = float)
parser.add_argument('--nnearest', default = 13,  type = int)
parser.add_argument('--nneigh',   default = 80,  type = int, choices = [18,80])
parser.add_argument('--phantom',  default = 'brain', choices = ['brain','rod'])
parser.add_argument('--seed',     default = 0, type = int)
parser.add_argument('--asym',     default = 0, type = int, choices = [0,1])

#parser.add_argument('--delta_t', default = 5., type = float)
#parser.add_argument('--nechos',   default = 2,  type = int)
#parser.add_argument('--ncoils',   default = 1,   type = int)

args = parser.parse_args()

niter       = args.niter
n_outer     = args.n_outer
bet_recon   = args.bet_recon
bet_gam     = args.bet_gam
n           = args.n
noise_level = args.noise_level
nnearest    = args.nnearest 
nneigh      = args.nneigh
phantom     = args.phantom
seed        = args.seed
asym        = args.asym
method      = args.method

ncoils      = 1
nechos      = 2
delta_t     = 5. / (nechos - 1)

odir = os.path.join('data','recons_multi_3d', '__'.join([x[0] + '_' + str(x[1]) for x in args.__dict__.items()]))

if not os.path.exists(odir):
    os.makedirs(odir)

# write input arguments to file
with open(os.path.join(odir,'input_params.csv'), 'w') as f:
  for x in args.__dict__.items():
    f.write("%s,%s\n"%(x[0],x[1]))

#--------------------------------------------------------------
#--------------------------------------------------------------

np.random.seed(seed)
  
#-------------------
# simulate images
#-------------------

if phantom == 'brain':

  label_nii = nib.load('data/brainweb54/subject54_crisp_v.mnc.gz')
  label_nii = nib.as_closest_canonical(label_nii)
  
  # pad to 512x512x512 voxels
  lab       = np.pad(label_nii.get_fdata(), ((36,36),(0,0),(36,36)),'constant')

  t1_nii = nib.load('data/brainweb54/subject54_t1w_p4.mnc.gz')
  t1_nii = nib.as_closest_canonical(t1_nii)
  t1     = t1_nii.get_fdata()

  lab_affine = label_nii.affine.copy()
  lab_affine[0,-1] -= 36*lab_affine[0,0]
  lab_affine[2,-1] -= 36*lab_affine[2,2]

  # CSF = 1, GM = 2, WM = 3
  csf_inds = np.where(lab == 1) 
  gm_inds  = np.where(lab == 2)
  wm_inds  = np.where(lab == 3)
  
  # set up array for trans. magnetization
  f = np.zeros(lab.shape)
  f[csf_inds] = 1.1
  f[gm_inds]  = 0.9
  f[wm_inds]  = 0.7

  # set up array for Gamma (ration between 2nd and 1st echo
  Gam = np.ones(f.shape)
  Gam[lab == 1] = np.exp(-delta_t/50)
  Gam[lab == 2] = 0.6*np.exp(-delta_t/8) + 0.4*np.exp(-delta_t/15)
  Gam[lab == 3] = 0.6*np.exp(-delta_t/9) + 0.4*np.exp(-delta_t/18)

  # set up image for special ROIs
  roi_vol = np.zeros(f.shape, dtype = np.int8)

  # add region with lower Gamma due to BO inhomogeneity
  x = np.linspace(-1,1,Gam.shape[0])
  X,Y,Z = np.meshgrid(x,x,x, indexing = 'ij')
  R = np.sqrt((X + 0.15)**2 + (Y - 0.7)**2 + (Z - 0)**2)

  B0weights = gaussian_filter((R<0.15).astype(float), 3)
  Gam = (1-B0weights)*Gam + 0.4*B0weights

  roi_vol[R<0.15] = 1

  # add stand alone lesion in Na density
  RL = np.sqrt((X - 0.2)**2 + (Y - 0.4)**2 + (Z - 0)**2)
  Lweights = gaussian_filter((RL<0.05).astype(float), 1.)
  f = (1-Lweights)*f + 1.*Lweights

  roi_vol[RL<0.05] = 2

  # interpolate T1 to label grid
  t1 = aff_transform(t1, np.linalg.inv(t1_nii.affine) @ lab_affine,
                     f.shape, cval = t1.min()) 

  # add lesion in t1
  RT = np.sqrt((X + 0.367)**2 + (Y + 0.372)**2 + (Z - 0)**2)
  Tweights = gaussian_filter((RT<0.05).astype(float), 1.)
  t1 = (1-Tweights)*t1 + 0.1*Tweights

  roi_vol[RT<0.05] = 3

  # regrid to 128x128x128 or 256x256x256 voxels
  f   = zoom3d(f,  n/f.shape[0])
  t1  = zoom3d(t1, n/t1.shape[0])
  Gam = zoom3d(Gam, n/Gam.shape[0])

  roi_vol = zoom(roi_vol, n/Gam.shape[0], order = 0, prefilter = False)

  # reorient to LPS
  f   = np.flip(f,(0,1))
  t1  = np.flip(t1,(0,1))
  Gam = np.flip(Gam,(0,1))
  roi_vol = np.flip(roi_vol,(0,1))

elif phantom == 'rod':
  n4 = 4*n

  x = np.arange(n4) - n4/2 + 0.5
  X,Y,Z = np.meshgrid(x, x, x, indexing = 'ij')

  # set up sodium content
  f = np.zeros((n4,n4,n4))
  f[np.sqrt(X**2 + Y**2) <= 0.4*n4] = 1

  # add rods
  f[np.sqrt((X-(n4/4))**2 + Y**2) <= 0.03*n4] = 2
  f[np.sqrt((X+(n4/4))**2 + Y**2) <= 0.03*n4] = 0
  f[np.sqrt((X-(n4/8))**2 + Y**2) <= 0.01*n4] = 2
  f[np.sqrt((X+(n4/8))**2 + Y**2) <= 0.01*n4] = 0

  f[np.sqrt((Y-(n4/4))**2 + X**2) <= 0.03*n4] = 2
  f[np.sqrt((Y+(n4/4))**2 + X**2) <= 0.03*n4] = 0
  f[np.sqrt((Y-(n4/8))**2 + X**2) <= 0.01*n4] = 2
  f[np.sqrt((Y+(n4/8))**2 + X**2) <= 0.01*n4] = 0

  f[np.abs(Z) > 0.3*n4] = 0

  # down sample array
  f = f.reshape(n,4,n,4,n,4).mean(axis=(1,3,5))

  # set up sodium content
  Gam = np.ones((n4,n4,n4))
  Gam[np.sqrt(X**2 + Y**2) <= 0.4*n4] = np.exp(-delta_t/50)

  # add rods
  Gam[np.sqrt((X-(n4/4))**2 + Y**2) <= 0.03*n4] = np.exp(-delta_t/50)
  Gam[np.sqrt((X+(n4/4))**2 + Y**2) <= 0.03*n4] = np.exp(-delta_t/50)
  Gam[np.sqrt((X-(n4/8))**2 + Y**2) <= 0.01*n4] = np.exp(-delta_t/50)
  Gam[np.sqrt((X+(n4/8))**2 + Y**2) <= 0.01*n4] = np.exp(-delta_t/50)

  Gam[np.sqrt((Y-(n4/4))**2 + X**2) <= 0.03*n4] = 0.6*np.exp(-delta_t/8) + 0.4*np.exp(-delta_t/15)
  Gam[np.sqrt((Y+(n4/4))**2 + X**2) <= 0.03*n4] = 0.6*np.exp(-delta_t/8) + 0.4*np.exp(-delta_t/15)
  Gam[np.sqrt((Y-(n4/8))**2 + X**2) <= 0.01*n4] = 0.6*np.exp(-delta_t/8) + 0.4*np.exp(-delta_t/15)
  Gam[np.sqrt((Y+(n4/8))**2 + X**2) <= 0.01*n4] = 0.6*np.exp(-delta_t/8) + 0.4*np.exp(-delta_t/15)

  Gam[np.abs(Z) > 0.3*n4] = 1

  # down sample array
  Gam = Gam.reshape(n,4,n,4,n,4).mean(axis=(1,3,5))


f = np.stack((f,np.zeros(f.shape)), axis = -1)
abs_f = np.linalg.norm(f, axis = -1)

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

# set up the array for the coil sensitivities
# the coil sensitivies are complex
sens        = np.zeros((ncoils,n,n,n,2))
sens[...,0] = 1.

#------------
#------------

# calculate signal on GPU
signal = apodized_fft_multi_echo(cp.asarray(f), readout_inds, cp.asarray(Gam), tr, delta_t, nechos = nechos,
                                 sens = cp.asarray(sens)).get()

kmask  = np.zeros(signal.shape)
for j in range(ncoils):
  for i in range(nechos):
    kmask[j,i,...,0] = (read_out_img > 0).astype(np.float)
    kmask[j,i,...,1] = (read_out_img > 0).astype(np.float)

# add noise to signal
if noise_level > 0:
  signal += noise_level*(np.random.randn(*signal.shape))*np.sqrt(nechos)/np.sqrt(2)

# multiply signal with readout mask
signal *= kmask
abs_signal = np.linalg.norm(signal, axis = -1)

ifft          = np.zeros((ncoils,nechos,) + f.shape)
ifft_filtered = np.zeros((ncoils,nechos,) + f.shape)
abs_ifft          = np.zeros((ncoils,nechos,) + f.shape[:-1])
abs_ifft_filtered = np.zeros((ncoils,nechos,) + f.shape[:-1])

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
aimg  = t1.copy()

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

Gam_recon = abs_ifft_filtered[:,1,...].mean(0) / abs_ifft_filtered[:,0,...].mean(0)
Gam_recon[Gam_recon > 1] = 1

# division by 3 is to compensate for norm of adjoint operator
recon = ifft_filtered[:,0,...].mean(0)
recon_shape = recon.shape
abs_recon   = np.linalg.norm(recon,axis=-1)

Gam_bounds = (n**3)*[(0.001,1)]

cost = []

fig1, ax1 = py.subplots(4, 7, figsize = (7*3,4*3), constrained_layout = True)
vmax = 1.5*abs_f.max()
ax1[0,0].imshow(abs_recon[...,n//2].T, vmin = 0, vmax = vmax, cmap = py.cm.Greys_r)
ax1[1,0].imshow(abs_recon[...,n//2].T - abs_f[...,n//2].T, vmax = 0.3, vmin = -0.3, cmap = py.cm.bwr)
ax1[2,0].imshow(Gam_recon[...,n//2].T, vmin = Gam.min(), vmax = 1, cmap = py.cm.Greys_r)
ax1[3,0].imshow(Gam_recon[...,n//2].T - Gam[...,n//2].T, vmin = -0.3, vmax = 0.3, cmap = py.cm.bwr)
ax1[0,0].set_title('init')

#--------------------------------------------------------------------------------------------------

for i in range(n_outer):

  print('LBFGS to optimize for recon')
  
  recon = recon.flatten()
  
  cb = lambda x: cost.append(multi_echo_bowsher_cost_total(x, recon_shape, signal, readout_inds, 
                             Gam_recon, tr, delta_t, nechos, kmask, bet_recon, bet_gam, ninds, method, sens, asym))
  
  res = fmin_l_bfgs_b(multi_echo_bowsher_cost,
                      recon, 
                      fprime = multi_echo_bowsher_grad, 
                      args = (recon_shape, signal, readout_inds, 
                              Gam_recon, tr, delta_t, nechos, kmask, bet_recon, ninds, ninds2, method, sens, asym),
                      callback = cb,
                      maxiter = niter, 
                      disp = 1)
  
  recon        = res[0].reshape(recon_shape)
  abs_recon    = np.linalg.norm(recon,axis=-1)
 
  if i % math.ceil(n_outer/6) == 0:
    im0_ax1 = ax1[0, (i//math.ceil(n_outer/6))+1].imshow(abs_recon[...,n//2].T, 
                                                         vmin = 0, vmax = vmax, cmap = py.cm.Greys_r)
    im1_ax1 = ax1[1, (i//math.ceil(n_outer/6))+1].imshow(abs_recon[...,n//2].T - abs_f[...,n//2].T, 
                                                         vmax = 0.3, vmin = -0.3, cmap = py.cm.bwr)
    ax1[0, (i//math.ceil(n_outer/6))+1].set_title(f'iter {i+1}')

  #---------------------------------------

  print('LBFGS to optimize for gamma')
  
  Gam_recon = Gam_recon.flatten()
  
  cb = lambda x: cost.append(multi_echo_bowsher_cost_total(recon, recon_shape, signal, readout_inds, 
                             x, tr, delta_t, nechos, kmask, bet_recon, bet_gam, ninds, method, sens, asym))
  
  res = fmin_l_bfgs_b(multi_echo_bowsher_cost_gamma,
                      Gam_recon, 
                      fprime = multi_echo_bowsher_grad_gamma, 
                      args = (recon_shape, signal, readout_inds, 
                              recon, tr, delta_t, nechos, kmask, bet_gam, ninds, ninds2, method, sens, asym),
                      callback = cb,
                      maxiter = niter, 
                      bounds = Gam_bounds,
                      disp = 1)
  
  Gam_recon = res[0].reshape(recon_shape[:-1])

  if i % math.ceil(n_outer/6) == 0:
    last_col = (i//math.ceil(n_outer/6))+1
    im2_ax1 = ax1[2, (i//math.ceil(n_outer/6))+1].imshow(Gam_recon[...,n//2].T, vmin = Gam.min(), 
                                                         vmax = 1, cmap = py.cm.Greys_r)
    im3_ax1 = ax1[3, (i//math.ceil(n_outer/6))+1].imshow(Gam_recon[...,n//2].T - Gam[...,n//2].T, 
                                                         vmin = -0.3, vmax = 0.3, cmap = py.cm.bwr)

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

for axx in ax1.flatten():
  axx.set_axis_off()

fig1.colorbar(im0_ax1, ax = ax1[0,last_col])
fig1.colorbar(im1_ax1, ax = ax1[1,last_col])
fig1.colorbar(im2_ax1, ax = ax1[2,last_col])
fig1.colorbar(im3_ax1, ax = ax1[3,last_col])

#fig1.tight_layout()
fig1.savefig(os.path.join(odir,'convergence.png'))
fig1.show()

# save the recons
output_file = os.path.join(odir, 'recons.h5')
with h5py.File(output_file, 'w') as hf:
  grp = hf.create_group('images')
  grp.create_dataset('Gam',           data = Gam)
  grp.create_dataset('Gam_recon',     data = Gam_recon)
  grp.create_dataset('ground_truth',  data = f)
  grp.create_dataset('signal',        data = signal)
  grp.create_dataset('recon',         data = recon)
  grp.create_dataset('roi_vol',       data = roi_vol)
  grp.create_dataset('ifft',          data = ifft)
  grp.create_dataset('ifft_filtered', data = ifft_filtered)
  grp.create_dataset('prior_image',   data = aimg)
  grp.create_dataset('cost',          data = cost)

#--------------------------------------------------------------------------------------------------

fig2, ax2 = py.subplots(3, 4, figsize = (5*3,3*3))

im00_ax2 = ax2[0,0].imshow(abs_f[...,n//2].T, vmin = 0, vmax = vmax, cmap = py.cm.Greys_r)
im01_ax2 = ax2[0,1].imshow(abs_ifft[0,0,...,n//2].T, vmin = 0, vmax = vmax, cmap = py.cm.Greys_r)
im02_ax2 = ax2[0,2].imshow(abs_ifft_filtered[0,0,...,n//2].T, vmin = 0, vmax = vmax, cmap = py.cm.Greys_r)
im03_ax2 = ax2[0,3].imshow(abs_recon[...,n//2].T, vmin = 0, vmax = vmax, cmap = py.cm.Greys_r)

im11_ax2 = ax2[1,1].imshow(abs_ifft[0,0,...,n//2].T - abs_f[...,n//2].T, vmin = -0.3, vmax = 0.3, cmap = py.cm.bwr)
im12_ax2 = ax2[1,2].imshow(abs_ifft_filtered[0,0,...,n//2].T - abs_f[...,n//2].T, vmin = -0.3, vmax = 0.3, cmap = py.cm.bwr)
im13_ax2 = ax2[1,3].imshow(abs_recon[...,n//2].T - abs_f[...,n//2].T, vmin = -0.3, vmax = 0.3, cmap = py.cm.bwr)

im20_ax2 = ax2[2,0].imshow(Gam[...,n//2].T, vmin = Gam.min(), vmax = 1, cmap = py.cm.Greys_r)
im21_ax2 = ax2[2,1].imshow(Gam_recon[...,n//2].T, vmin = Gam.min(), vmax = 1, cmap = py.cm.Greys_r)
im22_ax2 = ax2[2,2].imshow(Gam_recon[...,n//2].T - Gam[...,n//2].T, vmin = -0.3, vmax = 0.3, cmap = py.cm.bwr)
im23_ax2 = ax2[2,3].imshow(aimg[...,n//2].T, cmap = py.cm.Greys_r)

fig2.colorbar(im00_ax2, ax = ax2[0,0], shrink = 0.5)
fig2.colorbar(im01_ax2, ax = ax2[0,1], shrink = 0.5)
fig2.colorbar(im02_ax2, ax = ax2[0,2], shrink = 0.5)
fig2.colorbar(im03_ax2, ax = ax2[0,3], shrink = 0.5)

fig2.colorbar(im11_ax2, ax = ax2[1,1], shrink = 0.5)
fig2.colorbar(im12_ax2, ax = ax2[1,2], shrink = 0.5)
fig2.colorbar(im13_ax2, ax = ax2[1,3], shrink = 0.5)

fig2.colorbar(im20_ax2, ax = ax2[2,0], shrink = 0.5)
fig2.colorbar(im21_ax2, ax = ax2[2,1], shrink = 0.5)
fig2.colorbar(im22_ax2, ax = ax2[2,2], shrink = 0.5)
fig2.colorbar(im23_ax2, ax = ax2[2,3], shrink = 0.5)

ax2[0,0].set_title('ground truth Na', size = 'medium')
ax2[0,1].set_title('IFFT 1st echo', size = 'medium')
ax2[0,2].set_title('IFFT 1st echo filtered', size = 'medium')
ax2[0,3].set_title('joint reconstruction', size = 'medium')

ax2[1,1].set_title('bias IFFT 1st echo', size = 'medium')
ax2[1,2].set_title('bias IFFT 1st echo filtered', size = 'medium')
ax2[1,3].set_title('bias joint reconstruction', size = 'medium')

ax2[2,0].set_title('ground truth Gamma', size = 'medium')
ax2[2,1].set_title('joint reconstruction Gamma', size = 'medium')
ax2[2,2].set_title('bias joint reconstruction Gamma', size = 'medium')
ax2[2,3].set_title('anat. prior image', size = 'medium')

for axx in ax2.flatten():
  axx.set_axis_off()

fig2.tight_layout()
fig2.show()
fig2.savefig(os.path.join(odir,'results_2d.png'))


ims1 = 2*[{'cmap':py.cm.Greys_r}] + [{'cmap':py.cm.Greys_r, 'vmin':0, 'vmax':vmax}]
vi1 = pv.ThreeAxisViewer([abs_ifft[0,0,...], abs_ifft_filtered[0,0,...],abs_f], imshow_kwargs = ims1)
vi1.fig.savefig(os.path.join(odir,'fig1.png'))

ims2 = [{'cmap':py.cm.Greys_r, 'vmin':0, 'vmax':vmax}] + 2*[{'cmap':py.cm.Greys_r, 'vmin':0.5, 'vmax':1.}]
vi2  = pv.ThreeAxisViewer([abs_recon,Gam_recon, Gam], imshow_kwargs = ims2)
vi2.fig.savefig(os.path.join(odir,'fig2.png'))

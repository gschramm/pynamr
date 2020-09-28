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

from scipy.ndimage     import zoom, gaussian_filter
from scipy.interpolate import interp1d

from argparse import ArgumentParser

#--------------------------------------------------------------
#--------------------------------------------------------------

parser = ArgumentParser(description = '3D na mr dual echo simulation')
parser.add_argument('--niter',  default = 10, type = int)
parser.add_argument('--niter_last', default = 10, type = int)
parser.add_argument('--n_outer', default = 6, type = int)
parser.add_argument('--method', default = 0, type = int)
parser.add_argument('--bet_recon', default = 0.1, type = float)
parser.add_argument('--bet_gam', default = 0.3, type = float)
parser.add_argument('--delta_t', default = 5., type = float)
parser.add_argument('--n', default = 128, type = int)
parser.add_argument('--noise_level', default = 0.2,  type = float)
parser.add_argument('--nechos',   default = 2,  type = int)
parser.add_argument('--nnearest', default = 13,  type = int)
parser.add_argument('--nneigh',   default = 80,  type = int, choices = [18,80])
parser.add_argument('--ncoils',   default = 1,   type = int)

args = parser.parse_args()

niter       = args.niter
niter_last  = args.niter_last
n_outer     = args.n_outer
bet_recon   = args.bet_recon * args.nechos / 2
bet_gam     = args.bet_gam * args.nechos / 2
method      = args.method
n           = args.n
delta_t     = args.delta_t / (args.nechos - 1)
noise_level = args.noise_level
nechos      = args.nechos
nnearest    = args.nnearest 
nneigh      = args.nneigh
ncoils      = args.ncoils

odir = os.path.join('data','recons_multi_3d', '__'.join([x[0] + '_' + str(x[1]) for x in args.__dict__.items()]))

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
# simulate images
#-------------------

data = np.load('./data/54.npz')
t1     = data['arr_0']
labels = data['arr_1']
lab    = np.pad(labels, ((36,36),(0,0),(36,36)),'constant')

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
f          = zoom(np.expand_dims(f,-1),(n/434,n/434,n/434,1), order = 1, prefilter = False)[...,0]
lab_regrid = zoom(lab, (n/434,n/434,n/434), order = 0, prefilter = False) 

# set up array for T2* times
Gam = np.ones((n,n,n))
Gam[lab_regrid == 1] = np.exp(-delta_t/50)
Gam[lab_regrid == 2] = 0.6*np.exp(-delta_t/8) + 0.4*np.exp(-delta_t/15)
Gam[lab_regrid == 3] = 0.6*np.exp(-delta_t/9) + 0.4*np.exp(-delta_t/18)

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
aimg  = (f.max() - f[...,0])**0.8
aimg += 0.001*aimg.max()*np.random.random(aimg.shape)

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
Gam_recon[abs_ifft_filtered[:,1,...].mean(0) < 0.1*abs_ifft_filtered[:,0,...].mean(0).max()] = 1
Gam_recon[Gam_recon > 1] = 1

# division by 3 is to compensate for norm of adjoint operator
recon = ifft_filtered[:,0,...].mean(0)
recon_shape = recon.shape
abs_recon   = np.linalg.norm(recon,axis=-1)

Gam_bounds = (n**3)*[(0.001,1)]

cost = []

fig1, ax1 = py.subplots(4,n_outer+1, figsize = ((n_outer+1)*3,12))
vmax = 1.5*abs_f.max()
ax1[0,0].imshow(Gam_recon[...,64], vmin = 0, vmax = 1, cmap = py.cm.Greys_r)
ax1[1,0].imshow(Gam_recon[...,64] - Gam[...,64], vmin = -0.3, vmax = 0.3, cmap = py.cm.bwr)
ax1[2,0].imshow(abs_recon[...,64], vmin = 0, vmax = vmax, cmap = py.cm.Greys_r)
ax1[3,0].imshow(abs_recon[...,64] - abs_f[...,64], vmax = 0.3, vmin = -0.3, cmap = py.cm.bwr)

#--------------------------------------------------------------------------------------------------

for i in range(n_outer):

  print('LBFGS to optimize for recon')
  
  if i == (n_outer - 1):
    niter_recon = niter_last
  else:
    niter_recon = niter

  recon       = recon.flatten()
  
  cb = lambda x: cost.append(multi_echo_bowsher_cost_total(x, recon_shape, signal, readout_inds, 
                             Gam_recon, tr, delta_t, nechos, kmask, bet_recon, bet_gam, ninds, method, sens))
  
  res = fmin_l_bfgs_b(multi_echo_bowsher_cost,
                      recon, 
                      fprime = multi_echo_bowsher_grad, 
                      args = (recon_shape, signal, readout_inds, 
                              Gam_recon, tr, delta_t, nechos, kmask, bet_recon, ninds, ninds2, method, sens),
                      callback = cb,
                      maxiter = niter_recon, 
                      disp = 1)
  
  recon        = res[0].reshape(recon_shape)
  abs_recon    = np.linalg.norm(recon,axis=-1)
  
  ax1[2,i+1].imshow(abs_recon[...,64], vmin = 0, vmax = vmax, cmap = py.cm.Greys_r)
  ax1[3,i+1].imshow(abs_recon[...,64] - abs_f[...,64], vmax = 0.3, vmin = -0.3, cmap = py.cm.bwr)

  #---------------------------------------

  print('LBFGS to optimize for gamma')
  
  Gam_recon = Gam_recon.flatten()
  
  cb = lambda x: cost.append(multi_echo_bowsher_cost_total(recon, recon_shape, signal, readout_inds, 
                             x, tr, delta_t, nechos, kmask, bet_recon, bet_gam, ninds, method, sens))
  
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
  Gam_recon[abs_ifft_filtered[:,1,...].mean(0) < 0.1*abs_ifft_filtered[:,0,...].mean(0).max()] = 1

  ax1[0,i+1].imshow(Gam_recon[...,64], vmin = 0, vmax = 1, cmap = py.cm.Greys_r)
  ax1[1,i+1].imshow(Gam_recon[...,64] - Gam[...,64], vmin = -0.3, vmax = 0.3, cmap = py.cm.bwr)

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

fig1.tight_layout()
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
  grp.create_dataset('ifft',          data = ifft)
  grp.create_dataset('ifft_filtered', data = ifft_filtered)
  grp.create_dataset('prior_image',   data = aimg)
  grp.create_dataset('cost',          data = cost)

#--------------------------------------------------------------------------------------------------

import pymirc.viewer as pv

ims1 = 2*[{'cmap':py.cm.Greys_r}] + [{'cmap':py.cm.Greys_r, 'vmin':0, 'vmax':vmax}]
vi1 = pv.ThreeAxisViewer([abs_ifft[:,0,...].mean(0),abs_ifft_filtered[:,0,...].mean(0),abs_f], imshow_kwargs = ims1)
vi1.fig.savefig(os.path.join(odir,'fig1.png'))

ims2 = [{'cmap':py.cm.Greys_r, 'vmin':0, 'vmax':vmax}] + 2*[{'cmap':py.cm.Greys_r, 'vmin':0, 'vmax':1.}]
vi2  = pv.ThreeAxisViewer([abs_recon,Gam_recon, Gam], imshow_kwargs = ims2)
vi2.fig.savefig(os.path.join(odir,'fig2.png'))

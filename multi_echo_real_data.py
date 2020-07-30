import os
import numpy as np
import h5py
import nibabel as nib

import matplotlib as mpl
if os.getenv('DISPLAY') is None: mpl.use('Agg')
import matplotlib.pyplot as py

from scipy.optimize import fmin_l_bfgs_b, fmin_cg
from nearest_neighbors import nearest_neighbors, is_nearest_neighbor_of
from readout_time import readout_time
from apodized_fft import apodized_fft_multi_echo, adjoint_apodized_fft_multi_echo
from bowsher      import bowsher_prior_cost, bowsher_prior_grad

from scipy.ndimage     import zoom, gaussian_filter
from scipy.interpolate import interp1d

from argparse import ArgumentParser
#--------------------------------------------------------------
def multi_echo_data_fidelity(recon, signal, readout_inds, Gam, tr, delta_t, nechos, kmask):

  exp_data = apodized_fft_multi_echo(recon, readout_inds, Gam, tr, delta_t, nechos = nechos)
  diff     = (exp_data - signal)*kmask
  cost     = 0.5*(diff**2).sum()

  return cost

#--------------------------------------------------------------
def multi_echo_data_fidelity_grad(recon, signal, readout_inds, Gam, tr, delta_t, nechos, 
                                  kmask, grad_gamma):

  exp_data = apodized_fft_multi_echo(recon, readout_inds, Gam, tr, delta_t, nechos = nechos)
  diff     = (exp_data - signal)*kmask

  if grad_gamma:
    grad  = adjoint_apodized_fft_multi_echo(diff, readout_inds, Gam, tr, delta_t, grad_gamma = True)
    grad *= recon
  else:
    grad  = adjoint_apodized_fft_multi_echo(diff, readout_inds, Gam, tr, delta_t, grad_gamma = False)

  return grad

#--------------------------------------------------------------------
def multi_echo_bowsher_cost(recon, recon_shape, signal, readout_inds, Gam, tr, delta_t, nechos, kmask,
                           beta, ninds, ninds2, method):
  # ninds2 is a dummy argument to have the same arguments for
  # cost and its gradient

  isflat_recon = False
  isflat_Gam   = False

  if recon.ndim == 1:  
    isflat_recon = True
    recon  = recon.reshape(recon_shape)

  if Gam.ndim == 1:  
    isflat_Gam = True
    Gam  = Gam.reshape(recon_shape[:-1])

  cost = multi_echo_data_fidelity(recon, signal, readout_inds, Gam, tr, delta_t, nechos, kmask)

  if beta > 0:
    cost += beta*bowsher_prior_cost(recon[...,0], ninds, method)
    cost += beta*bowsher_prior_cost(recon[...,1], ninds, method)

  if isflat_recon:
    recon = recon.flatten()

  if isflat_Gam:
    Gam = Gam.flatten()
   
  return cost

#--------------------------------------------------------------------
def multi_echo_bowsher_cost_gamma(Gam, recon_shape, signal, readout_inds, recon, tr, delta_t, 
                                  nechos, kmask, beta, ninds, ninds2, method):
  return multi_echo_bowsher_cost(recon, recon_shape, signal, readout_inds, Gam, tr, delta_t, 
                                 nechos, kmask, beta, ninds, ninds2, method)

#--------------------------------------------------------------------
def multi_echo_bowsher_grad(recon, recon_shape, signal, readout_inds, Gam, tr, delta_t, nechos, kmask,
                           beta, ninds, ninds2, method):

  isflat = False
  if recon.ndim == 1:  
    isflat = True
    recon  = recon.reshape(recon_shape)

  grad = multi_echo_data_fidelity_grad(recon, signal, readout_inds, Gam, tr, delta_t, 
                                       nechos, kmask, False)

  if beta > 0:

    grad[...,0] += beta*bowsher_prior_grad(recon[...,0], ninds, ninds2, method)
    grad[...,1] += beta*bowsher_prior_grad(recon[...,1], ninds, ninds2, method)

  if isflat:
    recon = recon.flatten()
    grad  = grad.flatten()

  return grad

#--------------------------------------------------------------------
def multi_echo_bowsher_grad_gamma(Gam, recon_shape, signal, readout_inds, recon, tr, delta_t, nechos, 
                                  kmask, beta, ninds, ninds2, method):

  isflat = False
  if Gam.ndim == 1:  
    isflat = True
    Gam = Gam.reshape(recon_shape[:-1])

  tmp = multi_echo_data_fidelity_grad(recon, signal, readout_inds, Gam, tr, delta_t, nechos, kmask, True)

  grad = tmp[...,0] + tmp[...,1]

  if beta > 0:
    grad += beta*bowsher_prior_grad(Gam, ninds, ninds2, method)

  if isflat:
    Gam  = Gam.flatten()
    grad = grad.flatten()

  return grad


#--------------------------------------------------------------
#--------------------------------------------------------------
parser = ArgumentParser(description = '2D na mr dual echo simulation')
parser.add_argument('--niter',  default = 10, type = int)
parser.add_argument('--niter_last', default = 10, type = int)
parser.add_argument('--n_outer', default = 6, type = int)
parser.add_argument('--method', default = 0, type = int)
parser.add_argument('--bet_recon', default = 3, type = float)
parser.add_argument('--bet_gam', default = 10, type = float)
parser.add_argument('--nnearest', default = 3,   type = int)
parser.add_argument('--nneigh',   default = 20,  type = int, choices = [8,20])

args = parser.parse_args()

niter       = args.niter
niter_last  = args.niter_last
n_outer     = args.n_outer
bet_recon   = args.bet_recon
bet_gam     = args.bet_gam
method      = args.method
nnearest    = args.nnearest 
nneigh      = args.nneigh

odir = os.path.join('data','recons_multi_real', '__'.join([x[0] + '_' + str(x[1]) for x in args.__dict__.items()]))

if not os.path.exists(odir):
    os.makedirs(odir)

# write input arguments to file
with open(os.path.join(odir,'input_params.csv'), 'w') as f:
  for x in args.__dict__.items():
    f.write("%s,%s\n"%(x[0],x[1]))

n       = 128
delta_t = 5.
nechos  = 2
sl      = 54

py.rc('image', cmap='gray')
#--------------------------------------------------------------
#--------------------------------------------------------------

np.random.seed(0)
  
#-------------------
# load the data
#-------------------

# read the 3D sodium and aligned T1 volume in 128 grid

na_nii   = nib.load('./data/SodiumExample/na_128.nii')
na2_nii  = nib.load('./data/SodiumExample/na2_128.nii')
t1_nii   = nib.load('./data/SodiumExample/T1_128_aligned.nii')

na_nii  = nib.as_closest_canonical(na_nii)
na2_nii = nib.as_closest_canonical(na2_nii)
t1_nii  = nib.as_closest_canonical(t1_nii)

na_vol  = na_nii.get_fdata()
na2_vol = na2_nii.get_fdata()
t1_vol  = t1_nii.get_fdata()

f = na_vol[:,:,sl].astype(np.float64)
f = np.stack((f,np.zeros(f.shape)), axis = -1)

f2 = na2_vol[:,:,sl].astype(np.float64)
f2 = np.stack((f2,np.zeros(f2.shape)), axis = -1)

#-------------------
# calc readout times
#-------------------

# setup the frequency array as used in numpy fft
k0,k1 = np.meshgrid(np.arange(n) - n//2 + 0.5, np.arange(n) - n//2 + 0.5)
abs_k = np.sqrt(k0**2 + k1**2)
abs_k = np.fft.fftshift(abs_k)

# rescale abs_k such that k = 1.5 is at r = 32 (the edge)
k_edge = 1.5
abs_k *= k_edge/32

# calculate the readout times and the k-spaces locations that
# are read at a given time
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

signal = np.array([np.fft.fft2(f[:,:,0]).view('(2,)float'), np.fft.fft2(f2[:,:,0]).view('(2,)float')])

kmask  = np.ones(signal.shape)
#kmask  = np.zeros(signal.shape)
#for i in range(nechos):
#  kmask[i,...,0] = (read_out_img > 0).astype(np.float)
#  kmask[i,...,1] = (read_out_img > 0).astype(np.float)

# multiply signal with readout mask
signal *= kmask

#----------------------------------------------------------------------------------------
# --- set up stuff for the prior
aimg = t1_vol[:,:,sl]

# beta = 1e-4 reasonable for inverse crime
if nneigh == 8:
  s    = np.array([[1,1,1], 
                   [1,0,1], 
                   [1,1,1]])
  
elif nneigh == 20:
  s    = np.array([[0,1,1,1,0], 
                   [1,1,1,1,1], 
                   [1,1,0,1,1], 
                   [1,1,1,1,1], 
                   [0,1,1,1,0]])
 
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

Gam_bounds = (n**2)*[(0.001,1)]

cost1 = []
cost2 = []

py.rc('image', cmap='gray')
py.rcParams['text.usetex'] = True
py.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']
py.rcParams['font.family'] = 'sans-serif'
py.rcParams['font.sans-serif'] = 'Computer Modern Sans Serif'

fig1, ax1 = py.subplots(2,n_outer+1, figsize = ((n_outer+1)*3,6))
vmax = 1.1*np.linalg.norm(f, axis = -1).max()
ax1[0,0].imshow(Gam_recon, vmin = 0, vmax = 1)
ax1[1,0].imshow(abs_recon, vmin = 0, vmax = vmax)

#--------------------------------------------------------------------------------------------------

for i in range(n_outer):

  print('LBFGS to optimize for recon')
  
  if i == (n_outer - 1):
    niter_recon = niter_last
  else:
    niter_recon = niter

  recon       = recon.flatten()
  
  cb = lambda x: cost2.append(multi_echo_bowsher_cost(x, recon_shape, signal, readout_inds, 
                             Gam_recon, tr, delta_t, nechos, kmask, bet_recon, ninds, ninds2, method))
  
  res = fmin_l_bfgs_b(multi_echo_bowsher_cost,
                      recon, 
                      fprime = multi_echo_bowsher_grad, 
                      args = (recon_shape, signal, readout_inds, 
                              Gam_recon, tr, delta_t, nechos, kmask, bet_recon, ninds, ninds2, method),
                      callback = cb,
                      maxiter = niter_recon, 
                      disp = 1)
  
  recon        = res[0].reshape(recon_shape)
  abs_recon    = np.linalg.norm(recon,axis=-1)
  
  ax1[1,i+1].imshow(abs_recon, vmin = 0)

  #---------------------------------------

  print('LBFGS to optimize for gamma')
  
  Gam_recon = Gam_recon.flatten()
  
  cb = lambda x: cost1.append(multi_echo_bowsher_cost_gamma(x, recon_shape, signal, readout_inds, 
                             recon, tr, delta_t, nechos, kmask, bet_gam, ninds, ninds2, method))
  
  res = fmin_l_bfgs_b(multi_echo_bowsher_cost_gamma,
                      Gam_recon, 
                      fprime = multi_echo_bowsher_grad_gamma, 
                      args = (recon_shape, signal, readout_inds, 
                              recon, tr, delta_t, nechos, kmask, bet_gam, ninds, ninds2, method),
                      callback = cb,
                      maxiter = niter, 
                      bounds = Gam_bounds,
                      disp = 1)
  
  Gam_recon = res[0].reshape(recon_shape[:-1])

  # reset values in low signal regions
  Gam_recon[f[...,0] < 0.1*f[...,0].max()] = 1

  ax1[0,i+1].imshow(Gam_recon, vmin = 0, vmax = 1)

  #--------------------------------------------------------------------------------------------------

# normalize recon
norm = abs_recon.sum()/f[...,0].sum()
recon /= norm
abs_recon    = np.linalg.norm(recon,axis=-1)

# save the recons
output_file = os.path.join(odir, 'recons.h5')
with h5py.File(output_file, 'w') as hf:
  grp = hf.create_group('images')
  grp.create_dataset('Gam_recon',     data = Gam_recon)
  grp.create_dataset('signal',        data = signal)
  grp.create_dataset('recon',         data = recon)
  grp.create_dataset('prior_image',   data = aimg)

#--------------------------------------------------------------------------------------------------

fig1.tight_layout()
fig1.savefig(os.path.join(odir,'convergence.png'))
fig1.show()

fig,ax = py.subplots(1,4, figsize = (16,4))
ax[0].imshow(f[...,0].T, origin = 'lower', vmax = 1)
ax[1].imshow(aimg.T, origin = 'lower')
ax[2].imshow(abs_recon.T, origin = 'lower', vmax = 1)
ax[3].imshow(Gam_recon.T, origin = 'lower', vmax = 1)
for axx in ax: axx.set_axis_off()
fig.tight_layout()
fig.savefig(os.path.join(odir,'results.png'))
fig.show()



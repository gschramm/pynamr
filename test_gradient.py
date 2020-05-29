import os
import numpy as np
import h5py

import matplotlib as mpl
if os.getenv('DISPLAY') is None: mpl.use('Agg')
import matplotlib.pyplot as py

from scipy.optimize import fmin_l_bfgs_b, fmin_cg
from nearest_neighbors import nearest_neighbors, is_nearest_neighbor_of
from readout_time import readout_time
from apodized_fft import apodized_fft_multi_echo, adjoint_apodized_fft_multi_echo
from bowsher      import bowsher_prior_cost, bowsher_prior_grad

from scipy.ndimage     import zoom, gaussian_filter

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
  grad     = adjoint_apodized_fft_multi_echo(diff, readout_inds, Gam, tr, delta_t, 
                                             grad_gamma = grad_gamma)

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
parser.add_argument('--bet_recon', default = 0.5, type = float)
parser.add_argument('--bet_gam', default = 0.5, type = float)
parser.add_argument('--delta_t', default = 5., type = float)
parser.add_argument('--n', default = 128, type = int)
parser.add_argument('--noise_level', default = 0.1,  type = float)
parser.add_argument('--nechos', default = 2,  type = int)

args = parser.parse_args()

niter       = args.niter
niter_last  = args.niter_last
n_outer     = args.n_outer
bet_recon   = args.bet_recon
bet_gam     = args.bet_gam
method      = args.method
n           = args.n
delta_t     = args.delta_t / (args.nechos - 1)
noise_level = args.noise_level
nechos      = args.nechos

odir = os.path.join('data','recons_multi', '__'.join([x[0] + '_' + str(x[1]) for x in args.__dict__.items()]))

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
lab    = np.pad(labels[:,:,132].transpose(), ((0,0),(36,36)),'constant')

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
Gam = np.ones((n,n))
Gam[lab_regrid == 1] = np.exp(-delta_t/50)
Gam[lab_regrid == 2] = 0.6*np.exp(-delta_t/8) + 0.4*np.exp(-delta_t/15)
Gam[lab_regrid == 3] = 0.6*np.exp(-delta_t/9) + 0.4*np.exp(-delta_t/18)

f = np.stack((f,np.zeros(f.shape)), axis = -1)

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
#------------

signal = apodized_fft_multi_echo(f, readout_inds, Gam, tr, delta_t, nechos = nechos)

kmask  = np.zeros(signal.shape)
for i in range(nechos):
  kmask[i,...,0] = (read_out_img > 0).astype(np.float)
  kmask[i,...,1] = (read_out_img > 0).astype(np.float)

# multiply signal with readout mask
signal *= kmask
abs_signal = np.linalg.norm(signal, axis = -1)

# add noise to signal
if noise_level > 0:
  signal += noise_level*(np.random.randn(*signal.shape))*np.sqrt(nechos)/np.sqrt(2)

# multiply signal with readout mask
signal *= kmask
abs_signal = np.linalg.norm(signal, axis = -1)

# calculate the inverse FFT of the data for all echos
ifft_fac = np.sqrt(np.prod(f.shape)) / np.sqrt(4*(signal.ndim - 1))

ifft     = np.zeros((nechos,) + f.shape)
abs_ifft = np.zeros((nechos,) + f.shape[:-1])

for i in range(nechos):
  s = signal[i,...].view(dtype = np.complex128).squeeze().copy()
  ifft[i,...] = np.ascontiguousarray(np.fft.ifft2(s).view('(2,)float') * ifft_fac)
  abs_ifft[i,...] = np.linalg.norm(ifft[i,...], axis = -1)

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

#--------------------------------------------------------------------------------------------------
# initialize variables

g0 = gaussian_filter(abs_ifft[0,...],2)
g1 = gaussian_filter(abs_ifft[1,...],2)

Gam_recon =  g1 / g0
Gam_recon[abs_ifft[1,...] < 0.1*g0.max()] = 1
Gam_recon[Gam_recon > 1] = 1

recon = ifft[0,...].copy()
recon_shape = recon.shape
abs_recon   = np.linalg.norm(recon,axis=-1)

Gam_bounds = (n**2)*[(0.001,1)]

cost1 = []
cost2 = []

#### Hack test gradient
Gam1 = Gam.copy()
eps  = 1e-6

g1 = multi_echo_bowsher_grad_gamma(Gam1, recon_shape, signal, readout_inds,                           
                                   recon, tr, delta_t, nechos, kmask, bet_gam, ninds, ninds2, method)
g2 = np.zeros(g1.shape)

for i in range(n):
  print(i)
  for j in range(n):
    Gam2 = Gam.copy()
    Gam2[i,j] += eps
    c1 = multi_echo_bowsher_cost_gamma(Gam1, recon_shape, signal, readout_inds, 
                                       recon, tr, delta_t, nechos, kmask, bet_gam, ninds, ninds2, method)
    c2 = multi_echo_bowsher_cost_gamma(Gam2, recon_shape, signal, readout_inds, 
                                       recon, tr, delta_t, nechos, kmask, bet_gam, ninds, ninds2, method)
    g2[i,j] = (c2 - c1) / eps

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
parser.add_argument('--bet_recon', default = 3., type = float)
parser.add_argument('--bet_gam', default = 10., type = float)
parser.add_argument('--delta_t', default = 5., type = float)
parser.add_argument('--n', default = 128, type = int)
parser.add_argument('--noise_level', default = 0.1,  type = float)
parser.add_argument('--nechos', default = 2,  type = int)

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

# plot central line trough k-space
fig3, ax3 = py.subplots(3,nechos, figsize = (nechos*3,9))
for i in range(nechos):
  ax3[0,i].plot(np.fft.fftshift(signal[i,:,:,0])[n//2,(n//2 - 32):(n//2 + 32)], '-')
  ax3[0,i].set_title(f'echo {i+1} real part')
  ax3[1,i].plot(np.fft.fftshift(signal[i,:,:,1])[n//2,(n//2 - 32):(n//2 + 32)], '-')
  ax3[1,i].set_title(f'echo {i+1} imag part')
  ax3[2,i].plot(np.fft.fftshift(abs_signal[i,:,:])[n//2,(n//2 - 32):(n//2 + 32)], '-')
  ax3[2,i].set_title(f'echo {i+1} abs value')
for axx in ax3[:-1,:].flatten(): 
  axx.set_ylim(-1,1)
  axx.grid(ls = ':')
for axx in ax3[-1,:].flatten(): 
  axx.set_ylim(0,1)
  axx.grid(ls = ':')
fig3.tight_layout()

# add noise to signal
if noise_level > 0:
  signal += noise_level*(np.random.randn(*signal.shape))*np.sqrt(nechos)/np.sqrt(2)

# multiply signal with readout mask
signal *= kmask
abs_signal = np.linalg.norm(signal, axis = -1)

# replot noisy signal
for i in range(nechos):
  ax3[0,i].plot(np.fft.fftshift(signal[i,:,:,0])[n//2,(n//2 - 32):(n//2 + 32)], '.')
  ax3[1,i].plot(np.fft.fftshift(signal[i,:,:,1])[n//2,(n//2 - 32):(n//2 + 32)], '.')
  ax3[2,i].plot(np.fft.fftshift(abs_signal[i,:,:])[n//2,(n//2 - 32):(n//2 + 32)], '.')

# calculate the inverse FFT of the data for all echos
ifft_fac = np.sqrt(np.prod(f.shape)) / np.sqrt(4*(signal.ndim - 1))

ifft          = np.zeros((nechos,) + f.shape)
ifft_filtered = np.zeros((nechos,) + f.shape)
abs_ifft          = np.zeros((nechos,) + f.shape[:-1])
abs_ifft_filtered = np.zeros((nechos,) + f.shape[:-1])

# create the han window that we need to multiply to the mask
h_win = interp1d(np.arange(32), np.hanning(64)[32:], fill_value = 0, bounds_error = False)
# abs_k was scaled to have the k edge at 32, we have to revert that for the han window
hmask = h_win(abs_k.flatten()*32/k_edge).reshape(n,n)

for i in range(nechos):
  s = signal[i,...].view(dtype = np.complex128).squeeze().copy()
  ifft[i,...] = np.ascontiguousarray(np.fft.ifft2(s).view('(2,)float') * ifft_fac)
  ifft_filtered[i,...] = np.ascontiguousarray(np.fft.ifft2(hmask*s).view('(2,)float') * ifft_fac)
  abs_ifft[i,...] = np.linalg.norm(ifft[i,...], axis = -1)
  abs_ifft_filtered[i,...] = np.linalg.norm(ifft_filtered[i,...], axis = -1)

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

Gam_recon = abs_ifft_filtered[1,...] / abs_ifft_filtered[0,...]
Gam_recon[abs_ifft_filtered[1,...] < 0.1*abs_ifft_filtered[0,...].max()] = 1
Gam_recon[Gam_recon > 1] = 1

recon = ifft_filtered[0,...].copy()
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

fig1, ax1 = py.subplots(4,n_outer+1, figsize = ((n_outer+1)*3,12))
vmax = 1.1*np.linalg.norm(f, axis = -1).max()
ax1[0,0].imshow(Gam_recon, vmin = 0, vmax = 1)
ax1[1,0].imshow(Gam_recon - Gam, vmin = -0.2, vmax = 0.2, cmap = py.cm.bwr)
ax1[2,0].imshow(abs_recon, vmin = 0, vmax = vmax)
ax1[3,0].imshow(abs_recon - np.linalg.norm(f, axis = -1), vmax = 0.2, vmin = -0.2, cmap = py.cm.bwr)
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
  
  ax1[2,i+1].imshow(abs_recon, vmin = 0, vmax = vmax)
  ax1[3,i+1].imshow(abs_recon - np.linalg.norm(f, axis = -1), vmax = 0.2, vmin = -0.2, cmap = py.cm.bwr)

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
  Gam_recon[abs_ifft_filtered[1,...] < 0.1*abs_ifft_filtered[0,...].max()] = 1

  ax1[0,i+1].imshow(Gam_recon, vmin = 0, vmax = 1)
  ax1[1,i+1].imshow(Gam_recon - Gam, vmin = -0.2, vmax = 0.2, cmap = py.cm.bwr)

  #--------------------------------------------------------------------------------------------------
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

#--------------------------------------------------------------------------------------------------

fig1.tight_layout()
fig1.savefig(os.path.join(odir,'convergence.png'))
fig1.show()

fig, ax = py.subplots(2,5, figsize = (15,6))
ax[0,0].imshow(np.linalg.norm(f, axis = -1), vmax = vmax)
ax[0,0].set_title('ground truth signal')
ax[1,0].imshow(Gam, vmax = 1, vmin = 0)
ax[1,0].set_title(r'$\Gamma$')
ax[0,1].imshow(abs_ifft_filtered[0,...], vmax = vmax)
ax[0,1].set_title('filt. IFFT 1st echo')
ax[1,1].imshow(abs_ifft_filtered[-1,...], vmax = vmax)
ax[1,1].set_title('filt. IFFT last echo')
ax[0,2].imshow(abs_ifft[0,...], vmax = vmax)
ax[0,2].set_title('IFFT 1st echo')
ax[1,2].imshow(abs_ifft[-1,...], vmax = vmax)
ax[1,2].set_title('IFFT last echo')
ax[0,3].imshow(abs_recon, vmax = vmax)
ax[0,3].set_title(r'recon $\beta$ ' + f'{bet_recon}')
ax[1,3].imshow(Gam_recon, vmax = 1, vmin = 0)
ax[1,3].set_title(r'$\Gamma$ recon $\beta$ ' + f'{bet_gam}')
ax[0,4].imshow(abs_recon - np.linalg.norm(f, axis = -1), vmax = 0.2, vmin = -0.2, cmap = py.cm.bwr)
ax[0,4].set_title(r'bias recon $\beta$ ' + f'{bet_recon}')
ax[1,4].imshow(Gam_recon - Gam, vmax = 0.2, vmin = -0.2, cmap = py.cm.bwr)
ax[1,4].set_title(r'bias $\Gamma$ recon $\beta$ ' + f'{bet_gam}')
fig.tight_layout()
fig.savefig(os.path.join(odir,'results.png'))
fig.show()

fig2,ax2 = py.subplots(1,2, figsize = (6,3))
ax2[0].semilogy(cost1)
ax2[1].semilogy(cost2)
fig2.tight_layout()
fig2.savefig(os.path.join(odir,'cost.png'))
fig2.show()

fig3.show()
fig3.savefig(os.path.join(odir,'kspace_prof.png'))

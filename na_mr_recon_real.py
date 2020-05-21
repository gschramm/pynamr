# small demo script to verify implementation of discrete FT (with FFT)

import h5py
import os
import numpy as np
import matplotlib.pyplot as py
import nibabel as nib

from scipy.optimize import fmin_l_bfgs_b, fmin_cg

from   matplotlib.colors import LogNorm

from apodized_fft      import apodized_fft, apo_images
from nearest_neighbors import nearest_neighbors, is_nearest_neighbor_of
from readout_time      import readout_time

from cost_functions    import mr_bowsher_cost, mr_bowsher_grad

from scipy.ndimage     import zoom, gaussian_filter
from scipy.interpolate import interp1d


#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
# parse the command line
from argparse import ArgumentParser

parser = ArgumentParser(description = '2D na mr simulation')

parser.add_argument('--T2star_recon_short',  default =  8,   type = float)
parser.add_argument('--T2star_recon_long',   default = 15,   type = float)
parser.add_argument('--T2star_recon_csf',    default = 50,   type = float)
parser.add_argument('--beta',                default =  1,   type = float)

parser.add_argument('--niter',       default =  50,  type = int)
parser.add_argument('--method',      default =   0,  type = int, choices = [0,1])

args = parser.parse_args()

niter       = args.niter
beta        = args.beta
method      = args.method

save_recons  = True

T2star_recon_short = args.T2star_recon_short
T2star_recon_long  = args.T2star_recon_long
T2star_recon_csf   = args.T2star_recon_csf

n  = 128
sl = 54

#--------------------------------------------------------------------------------------

py.rc('image', cmap='gray')


# read the 3D sodium and aligned T1 volume in 128 grid

na_nii  = nib.load('./data/SodiumExample/na_128.nii')
t1_nii  = nib.load('./data/SodiumExample/T1_128_aligned.nii')
csf_nii = nib.load('./data/SodiumExample/csf_128_aligned.nii')

na_nii  = nib.as_closest_canonical(na_nii)
t1_nii  = nib.as_closest_canonical(t1_nii)
csf_nii = nib.as_closest_canonical(csf_nii)

na_vol  = na_nii.get_fdata()
t1_vol  = t1_nii.get_fdata()
csf_vol = csf_nii.get_fdata()

f = na_vol[:,:,sl].astype(np.float64)
f = np.stack((f,np.zeros(f.shape)), axis = -1)

#===========================================================================================

# create binary mask that shows the inner 64 cube of k-space that is read out
# needed for 0 filling

k0,k1 = np.meshgrid(np.arange(n) - n//2 + 0.5, np.arange(n) - n//2 + 0.5)

# the K-value of 1.5 corresponds to the edge of the cube with radius 32
k_edge = 1.5
abs_k  = np.sqrt(k0**2 + k1**2)
abs_k = np.fft.fftshift(abs_k)

kmask = (abs_k <= 32).astype(float)
kmask = np.stack((kmask,kmask), axis = -1)

# rescale abs_k such that k = 1.5 is at r = 32 (the edge)
abs_k *= 1.5/32

t_read_2d = 1000*readout_time(abs_k)

n_readout_bins = 32

k_1d = np.linspace(0, k_edge, n_readout_bins + 1)

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

#----------------------------------------------------------
#--- simulate the signal

signal = np.fft.fft2(f[:,:,0]).view('(2,)float')

# multiply signal with readout mas
signal *= kmask

#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
#--- do recons

csf_img  = csf_vol[:,:,sl]
csf_inds = np.where(csf_img > 0.75) 

T2star_short_recon = np.full((n,n), T2star_recon_short)
T2star_long_recon  = np.full((n,n), T2star_recon_long) 

#T2star_short_recon[csf_inds] = T2star_recon_csf
#T2star_long_recon[csf_inds]  = T2star_recon_csf

apo_imgs_recon = apo_images(t_read_1d, T2star_short_recon, T2star_long_recon)

init_recon = f.copy()

#----------------------------------------------------------------------------------------
# --- set up stuff for the prior
aimg = t1_vol[:,:,sl]

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

ifft = f[...,0]
ifft *= abs_bow_recon.mean() / ifft.mean()

vmax = 40

fig,ax = py.subplots(1,3, figsize = (12,4))
ax[0].imshow(f[...,0].T, origin = 'lower', vmax = vmax)
ax[1].imshow(aimg.T, origin = 'lower')
ax[2].imshow(abs_bow_recon.T, origin = 'lower', vmax = vmax)
for axx in ax: axx.set_axis_off()
fig.tight_layout()
fig.savefig(f'figs/real_data_beta_{beta}.png')
fig.show()


##-----------------------------------------------------------------------------------------
## save the recons
#if save_recons:
#  output_file = os.path.join('data','recons', '__'.join([x[0] + '_' + str(x[1]) for x in args.__dict__.items()]) + '.h5')
#  with h5py.File(output_file, 'w') as hf:
#    grp = hf.create_group('images')
#    grp.create_dataset('ifft_recon',   data = init_recon)
#    grp.create_dataset('noreg_recon',  data = noreg_recon)
#    grp.create_dataset('ground_truth', data = f)
#    grp.create_dataset('T2star_long',  data = T2star_long)
#    grp.create_dataset('T2star_short', data = T2star_short)
#    grp.create_dataset('T2star_long_recon',  data = T2star_long_recon)
#    grp.create_dataset('T2star_short_recon', data = T2star_short_recon)
#    if beta > 0:
#      grp.create_dataset('bow_recon',    data = bow_recon)

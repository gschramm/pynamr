import numpy as np
import matplotlib.pyplot as py

from readout_time import readout_time
from apodized_fft import apodized_fft_dual_echo, adjoint_apodized_fft_dual_echo

from scipy.ndimage     import zoom, gaussian_filter

np.random.seed(0)
  
n = 128
delta_t = 5.

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

signal = apodized_fft_dual_echo(f, readout_inds, Gam, tr, delta_t)

ifft_fac = np.sqrt(np.prod(f.shape)) / np.sqrt(4*(signal.ndim - 1))

ifft0 = np.fft.ifft2(signal[0,...].view(dtype = np.complex128).squeeze()).view('(2,)float') * ifft_fac
ifft1 = np.fft.ifft2(signal[1,...].view(dtype = np.complex128).squeeze()).view('(2,)float') * ifft_fac

#--------------------------------------------------------------------------------------------------
py.rc('image', cmap='gray')
py.rcParams['text.usetex'] = True
py.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']
py.rcParams['font.family'] = 'sans-serif'
py.rcParams['font.sans-serif'] = 'Computer Modern Sans Serif'
#py.rcParams['axes.titlesize'] =  'medium'

fig, ax = py.subplots(2,2, figsize = (7,7))
vmax = 1.1*np.linalg.norm(f, axis = -1).max()
ax[0,0].imshow(np.linalg.norm(f, axis = -1), vmax = vmax)
ax[0,0].set_title('ground truth signal')
ax[0,1].imshow(Gam, vmax = 1, vmin = 0)
ax[0,1].set_title(r'$\Gamma$')
ax[1,0].imshow(np.linalg.norm(ifft0, axis = -1), vmax = vmax)
ax[1,0].set_title('IFFT 1st echo')
ax[1,1].imshow(np.linalg.norm(ifft1, axis = -1), vmax = vmax)
ax[1,1].set_title('IFFT 2nd echo')
fig.tight_layout()
fig.show()

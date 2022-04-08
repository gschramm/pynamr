import os
import numpy as np
import nibabel as nib

from scipy.ndimage  import zoom
from scipy.optimize import minimize
from glob import glob

import pymirc.image_operations as pi
import pymirc.metrics as pm
import pymirc.viewer as pv

from scipy.ndimage import gaussian_filter
from argparse import ArgumentParser
import matplotlib.pyplot as py

parser = ArgumentParser()
parser.add_argument('case')
parser.add_argument('--sdir', default = 'DeNoise_kw0')
parser.add_argument('--n', default = 128, type = int)
args = parser.parse_args()

case = args.case
pdir = os.path.join('data','sodium_data',args.case)
sdir = args.sdir
n    = args.n

sdir1 = os.path.join(glob(os.path.join(pdir,'*TE03*'))[0], sdir.split('_')[0], sdir)
sdir2 = os.path.join(glob(os.path.join(pdir,'*TE5*'))[0], sdir.split('_')[0], sdir)
fpattern = '*.c?'

# create the output directory
odir = os.path.join(pdir,sdir + '_preprocessed')
if not os.path.exists(odir):
  os.makedirs(odir)

#----
# load the complex coil images for first echo
data_shape  = (64,64,64)
recon_shape = (n,n,n) 

fnames  = sorted(glob(os.path.join(sdir1, fpattern)))
fnames2 = sorted(glob(os.path.join(sdir2, fpattern)))
ncoils  = len(fnames)

if (len(fnames) != 8) or (len(fnames2) != 8):
  raise ValueError('Not enough data files found')

data           = np.zeros((ncoils,) + recon_shape,  dtype = np.complex64)
data_filt      = np.zeros((ncoils,) + recon_shape,  dtype = np.complex64)
cimg           = np.zeros((ncoils,) + data_shape,   dtype = np.complex64)
cimg_pad       = np.zeros((ncoils,) + recon_shape,  dtype = np.complex64)
sens           = np.zeros((ncoils,) + recon_shape,  dtype = np.complex64)
cimg_pad_filt  = np.zeros((ncoils,) + recon_shape,  dtype = np.complex64)

data2           = np.zeros((ncoils,) + recon_shape,  dtype = np.complex64)
data2_filt      = np.zeros((ncoils,) + recon_shape,  dtype = np.complex64)
cimg2           = np.zeros((ncoils,) + data_shape,   dtype = np.complex64)
cimg2_pad       = np.zeros((ncoils,) + recon_shape,  dtype = np.complex64)
cimg2_pad_filt  = np.zeros((ncoils,) + recon_shape,  dtype = np.complex64)

# calculate filter for data used to estimated sensitivity
k = np.fft.fftfreq(recon_shape[0])
k0,k1,k2 = np.meshgrid(k, k, k, indexing = 'ij')
abs_k    = np.sqrt(k0**2 + k1**2 + k2**2) 
filt     = np.exp(-(abs_k**2)/(2*0.02**2))

for i in range(ncoils):
  #---- load data of first echo
  # load complex image
  cimg[i,...] = np.flip(np.fromfile(fnames[i], dtype = np.complex64).reshape(data_shape).swapaxes(0,2), (1,2))
  
  # perform fft to get into k-space
  cimg_fft = np.fft.fftn(cimg[i,...], norm = 'ortho')

  # pad data with 0s
  data[i,...] = np.fft.fftshift(np.pad(np.fft.fftshift(cimg_fft), (recon_shape[0] - data_shape[0])//2))
  data_filt[i,...] = filt*data[i,...]

  # do inverse FFT of data
  cimg_pad[i,...] = np.fft.ifftn(data[i,...], norm = 'ortho')
  cimg_pad_filt[i,...] = np.fft.ifftn(data_filt[i,...], norm = 'ortho')

  # load data of 2nd echo
  cimg2[i,...] = np.flip(np.fromfile(fnames2[i], dtype = np.complex64).reshape(data_shape).swapaxes(0,2), (1,2))
  cimg2_fft    = np.fft.fftn(cimg2[i,...], norm = 'ortho')
  data2[i,...] = np.fft.fftshift(np.pad(np.fft.fftshift(cimg2_fft), (recon_shape[0] - data_shape[0])//2))
  data2_filt[i,...]     = filt*data2[i,...]
  cimg2_pad[i,...]      = np.fft.ifftn(data2[i,...], norm = 'ortho')
  cimg2_pad_filt[i,...] = np.fft.ifftn(data2_filt[i,...], norm = 'ortho')
 
#-------

# calculate the sum of square image
sos = np.sqrt((np.abs(cimg_pad)**2).sum(axis = 0))
sos_filt = np.sqrt((np.abs(cimg_pad_filt)**2).sum(axis = 0))

for i in range(ncoils):
  sens[i,...]  = cimg_pad_filt[i,...] / sos_filt

#vi = pv.ThreeAxisViewer([np.abs(sens),np.abs(cimg_pad)])

# save the data and the sensitities
np.save(os.path.join(odir,f'echo1_{recon_shape[0]}.npy'), data)
np.save(os.path.join(odir,f'echo2_{recon_shape[0]}.npy'), data2)
np.save(os.path.join(odir,f'sens_{recon_shape[0]}.npy'), sens)

########################
# show the sensitivity

for sl in [45]:
  fig, ax = py.subplots(3,3, figsize = (8,8))
  for i in range(8):
    ax.flatten()[i].imshow(np.abs(sens[i,:,:,sl]).T, origin = 'lower', 
                           cmap = py.cm.Greys_r, vmin = 0, vmax = 0.4)
    ax.flatten()[i].set_title(os.path.basename(fnames[i]))
  
  ax.flatten()[8].imshow(np.abs(sos[:,:,sl]).T, origin = 'lower', cmap = py.cm.Greys_r)
  ax.flatten()[8].set_title('sum(abs(coil imgs))')
  fig.tight_layout()
  fig.show()

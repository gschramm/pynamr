import os
import numpy as np
import nibabel as nib

from scipy.ndimage  import zoom
from scipy.optimize import minimize
from glob import glob

import pymirc.image_operations as pi
import pymirc.metrics as pm
import pymirc.viewer as pv

pdir = os.path.join('data','sodium_data','TBI-n005')
sdir = 'DeNoise_kw0'
n    = 128

sdir1 = os.path.join(glob(os.path.join(pdir,'*TE03'))[0], sdir.split('_')[0], sdir.split('_')[1])
sdir2 = os.path.join(glob(os.path.join(pdir,'*TE5'))[0], sdir.split('_')[0], sdir.split('_')[1])
fpattern = '*.c?'

t1_nii = nib.load(os.path.join(pdir,'mprage.nii'))
t1_nii = nib.as_closest_canonical(t1_nii)
t1     = t1_nii.get_fdata()
t1_affine = t1_nii.affine

csf_nii = nib.load(os.path.join(pdir,'c3mprage.nii'))
csf_nii = nib.as_closest_canonical(csf_nii)
csf     = csf_nii.get_fdata()
csf_affine = csf_nii.affine

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
sos = np.abs(cimg_pad).sum(axis = 0)
sos_filt = np.abs(cimg_pad_filt).sum(axis = 0)

for i in range(ncoils):
  sens[i,...] = cimg_pad_filt[i,...] / sos_filt

# save the data and the sensitities
np.save(os.path.join(odir,f'echo1_{recon_shape[0]}.npy'), data)
np.save(os.path.join(odir,f'echo2_{recon_shape[0]}.npy'), data2)
np.save(os.path.join(odir,f'sens_{recon_shape[0]}.npy'), sens)

#------------------------------------------------------------------------------
#- align T1--------------------------------------------------------------------
#------------------------------------------------------------------------------

fov = 223.
na_voxsize = fov / np.array(recon_shape)
na_affine = np.diag(np.concatenate((na_voxsize,[1])))
na_affine[:-1,-1] = -fov/2.

csf_coreg, coreg_aff, coreg_params = pi.rigid_registration(csf, sos, t1_affine, na_affine)

t1_coreg = pi.aff_transform(t1, coreg_aff, recon_shape, cval = t1.min())

## save the data
np.save(os.path.join(odir,f't1_coreg_{recon_shape[0]}.npy'), t1_coreg)
np.save(os.path.join(odir,f'csf_coreg_{recon_shape[0]}.npy'), csf_coreg)

##nib.save(nib.Nifti1Image(csf_na_grid, na_affine), './data/SodiumExample/csf_128_aligned.nii')
#nib.save(nib.Nifti1Image(t1_na_grid, na_affine), os.path.join(pdir,f'mprage_{n}_aligned.nii'))
#nib.save(nib.Nifti1Image(na_interp, na_affine), os.path.join(pdir,f'TE03_{n}_' + sdir + '.nii'))
#nib.save(nib.Nifti1Image(na2_interp, na_affine), os.path.join(pdir,f'TE5_{n}_' + sdir + '.nii'))
# 
#np.savetxt(os.path.join(pdir,'affine_{n}.txt'), regis_aff)

########################
import pymirc.viewer as pv
vi = pv.ThreeAxisViewer([t1_coreg, sos])

########################
# show the sensitivity

import matplotlib.pyplot as py
#for sl in [35,40,45,50,55]:
for sl in [35,45,55,65]:
  fig, ax = py.subplots(3,3, figsize = (8,8))
  for i in range(8):
    ax.flatten()[i].imshow(np.abs(sens[i,:,:,sl]).T, origin = 'lower', 
                           cmap = py.cm.Greys_r, vmin = 0, vmax = 0.4)
    ax.flatten()[i].set_title(os.path.basename(fnames[i]))
  
  ax.flatten()[8].imshow(np.abs(sos[:,:,sl]).T, origin = 'lower', cmap = py.cm.Greys_r)
  ax.flatten()[8].set_title('sum(abs(coil imgs))')
  fig.tight_layout()
  fig.show()

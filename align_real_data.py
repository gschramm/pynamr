import os
import numpy as np
import nibabel as nib

from scipy.ndimage  import zoom
from scipy.optimize import minimize
from glob import glob

import pymirc.image_operations as pi
import pymirc.metrics as pm
import pymirc.viewer as pv

pdir = os.path.join('data','sodium_data','BT-007_visit2')
sdir = 'PhyCha_kw1'
n    = 256

sdir1 = os.path.join(glob(os.path.join(pdir,'*TE03'))[0], sdir.split('_')[0], sdir)
sdir2 = os.path.join(glob(os.path.join(pdir,'*TE5'))[0], sdir.split('_')[0], sdir)
fpattern = '*.c?'

t1_nii = nib.load(os.path.join(pdir,'mprage.nii'))
t1_nii = nib.as_closest_canonical(t1_nii)
t1     = t1_nii.get_fdata()
t1_affine = t1_nii.affine

csf_nii = nib.load(os.path.join(pdir,'c3mprage.nii'))
csf_nii = nib.as_closest_canonical(csf_nii)
csf     = csf_nii.get_fdata()
csf_affine = csf_nii.affine

#----
# load the complex coil images for first echo
data_shape  = (64,64,64)
#recon_shape = (256,256,256) 
recon_shape = (128,128,128) 

fnames  = glob(os.path.join(sdir1, fpattern))
ncoils  = len(fnames)
data    = np.zeros((ncoils,) + recon_shape,  dtype = np.complex64)
data_filt = np.zeros((ncoils,) + recon_shape,  dtype = np.complex64)
cimg      = np.zeros((ncoils,) + data_shape,   dtype = np.complex64)
cimg_pad  = np.zeros((ncoils,) + recon_shape,  dtype = np.complex64)
sens      = np.zeros((ncoils,) + recon_shape,  dtype = np.complex64)
cimg_pad_filt  = np.zeros((ncoils,) + recon_shape,  dtype = np.complex64)

# calculate filter for data used to estimated sensitivity
k = np.fft.fftfreq(recon_shape[0])
k0,k1,k2 = np.meshgrid(k, k, k, indexing = 'ij')
abs_k    = np.sqrt(k0**2 + k1**2 + k2**2) 
filt     = np.exp(-(abs_k**2)/(2*0.02**2))

for i in range(ncoils):
  # load complex image
  cimg[i,...] = np.flip(np.fromfile(fnames[i], dtype = np.complex64).reshape(data_shape).swapaxes(0,2), (1,2))
  
  # perform fft to get into k-space
  cimg_fft = np.fft.fftn(cimg[i,...], norm = 'ortho')

  # pad data with 0s
  data[i,...] = np.fft.fftshift(np.pad(np.fft.fftshift(cimg_fft), 
                                (recon_shape[0] - data_shape[0])//2))
  data_filt[i,...] = filt*data[i,...]

  # do inverse FFT of data
  cimg_pad[i,...] = np.fft.ifftn(data[i,...], norm = 'ortho')
  cimg_pad_filt[i,...] = np.fft.ifftn(data_filt[i,...], norm = 'ortho')

#-------

# calculate the sum of square image
sos = np.abs(cimg_pad).sum(axis = 0)
sos_filt = np.abs(cimg_pad_filt).sum(axis = 0)

for i in range(ncoils):
  sens[i,...] = cimg_pad_filt[i,...] / sos_filt


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
#
##nib.save(nib.Nifti1Image(csf_na_grid, na_affine), './data/SodiumExample/csf_128_aligned.nii')
#nib.save(nib.Nifti1Image(t1_na_grid, na_affine), os.path.join(pdir,f'mprage_{n}_aligned.nii'))
#nib.save(nib.Nifti1Image(na_interp, na_affine), os.path.join(pdir,f'TE03_{n}_' + sdir + '.nii'))
#nib.save(nib.Nifti1Image(na2_interp, na_affine), os.path.join(pdir,f'TE5_{n}_' + sdir + '.nii'))
# 
#np.savetxt(os.path.join(pdir,'affine_{n}.txt'), regis_aff)

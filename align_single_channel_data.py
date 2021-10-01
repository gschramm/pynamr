import os
import numpy as np
import nibabel as nib

from scipy.ndimage  import zoom
from scipy.optimize import minimize
from glob import glob

import pymirc.image_operations as pi
import pymirc.metrics as pm
import pymirc.viewer as pv

pdir     = os.path.join('data','sodium_data','Sodium_single_coil_Subject_013021')
mr1_name = 'MPRAGE_regular_coil'
mr2_name = 'MPRAGE_sodium_coil'

#pdir     = os.path.join('data','sodium_data','Sodium_single_coil_Subject_020621')
#mr1_name = 'MPRAGE_regular_coil'
#mr2_name = 'FLAIR_regular_coil'

na_fov    = 223
n         = 128

#----------------------------------------------------------------------------------------------------

# load the first echo
cimg1  = np.fromfile(os.path.join(pdir,'Na_TE05.cstack'), dtype = np.complex64).reshape((n,n,n))
# load the second echo
cimg2  = np.fromfile(os.path.join(pdir,'Na_TE5.cstack'), dtype = np.complex64).reshape((n,n,n))

# load the reference MR in LPS
mr1_nii = nib.load(os.path.join(pdir,mr1_name,'Image_reoriented.nii'))
mr1     = mr1_nii.get_fdata()

csf1_nii = nib.load(os.path.join(pdir,mr1_name,'c3Image_reoriented.nii'))
csf1     = csf1_nii.get_fdata()

# flip data sets to get them in LPS orientation (nii header tags are wrong ...)
cimg1 = np.flip(np.swapaxes(cimg1,0,2),(0,1,2))
cimg2 = np.flip(np.swapaxes(cimg2,0,2),(0,1,2))

mr1  = np.flip(mr1,(0,1))
csf1 = np.flip(csf1,(0,1))

# coregister MR1 to cimg1
na_affine = csf1_nii.affine.copy()
na_affine[0,0] = na_fov/cimg1.shape[0]
na_affine[1,1] = na_fov/cimg1.shape[1]
na_affine[2,2] = na_fov/cimg1.shape[1]

csf1_coreg, coreg_aff, coreg_params = pi.rigid_registration(csf1, np.abs(cimg1),csf1_nii.affine,na_affine)
mr1_coreg = pi.aff_transform(mr1, coreg_aff, cimg1.shape, cval = mr1.min())


# generate data as FFT of complex Na images and constant sensitivity
data1 = np.expand_dims(np.fft.fftshift(np.fft.fftn(cimg1, norm = 'ortho')),0)
data2 = np.expand_dims(np.fft.fftshift(np.fft.fftn(cimg2, norm = 'ortho')),0)
sens  = np.ones((1,) + cimg1.shape, dtype = np.complex64)

#---------------------------------------------------------------------------------------

odir = os.path.join(pdir,'preprocessed')

# save the results
np.save(os.path.join(odir,f'echo1_{n}.npy'), data1)
np.save(os.path.join(odir,f'echo2_{n}.npy'), data2)
np.save(os.path.join(odir,f'sens_{n}.npy'), sens)

np.save(os.path.join(odir,f'{mr1_name}_coreg_{n}.npy'), mr1_coreg)
np.save(os.path.join(odir,f'{mr1_name}_csf_{n}.npy'), csf1_coreg)

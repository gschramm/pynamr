import os
import numpy as np
import nibabel as nib

from scipy.ndimage  import zoom
from scipy.optimize import minimize
from glob import glob

import pymirc.image_operations as pi
import pymirc.metrics as pm

#def read_read_data(sdir = './data/SodiumExample/TBI-n005_tpiRecon_FID125462_TE03/PhyCha/kw1', fpattern = '*.c?'): 
sdir = './data/SodiumExample/TBI-n005_tpiRecon_FID125462_TE03/PhyCha/kw1'
fpattern = '*.c?'

t1_nii = nib.load('./data/SodiumExample/T1.nii')
t1_nii = nib.as_closest_canonical(t1_nii)
t1     = t1_nii.get_fdata()
t1_affine = t1_nii.affine

csf_nii = nib.load('./data/SodiumExample/c3T1.nii')
csf_nii = nib.as_closest_canonical(csf_nii)
csf     = csf_nii.get_fdata()

#----
# load the complex coil images

fnames = glob(os.path.join(sdir, fpattern))

ncoils = len(fnames)

data_shape = (64,64,64)
data = np.zeros((ncoils,) + data_shape, dtype = np.complex64)


for i, fname in enumerate(fnames):
  data[i,...] = np.flip(np.fromfile(fname, dtype = np.complex64).reshape(data_shape).swapaxes(0,2), (0,2))
  
na = np.flip(np.abs(data).sum(axis = 0), (0,1))
na /= na.max()

#s2 = np.abs(data.sum(axis = 0))

#s3 = np.flip(np.fromfile(os.path.join(sdir,'tpirec.csos'), 
#                         dtype = np.complex64).reshape(data_shape).swapaxes(0,2), (0,2))

#f_3d   = s1.copy()
#f      = f_3d[:,:,26]
#signal = np.fft.fft2(f).astype(np.complex128)
#signal = signal.view('(2,)float')
#f = np.stack((f,np.zeros(f.shape)), axis = -1)

#----------------------------------------------------------------------
# align the T1 to the sodium image

# interpolate sodium image to a 256 grid
na_interp = zoom(na, 2, order = 1, prefilter = False) / na.max()

fov = 223.

na_voxsize = fov / np.array(na_interp.shape)

na_affine = np.diag(np.concatenate((na_voxsize,[1])))
na_affine[:-1,-1] = -fov/2.

# this is the affine that maps from the T1 onto the Na grid
pre_affine = np.linalg.inv(t1_affine) @ na_affine

reg_params = np.zeros(6)
res = minimize(pm.regis_cost_func, reg_params, 
               args = (na_interp, csf, True, True, pm.neg_mutual_information, pre_affine), 
               method = 'Powell', 
               options = {'ftol':1e-2, 'xtol':1e-2, 'disp':True, 'maxiter':20, 'maxfev':5000})

reg_params = res.x.copy()

regis_aff = pre_affine @ pi.kul_aff(reg_params, origin = np.array(na_interp.shape)/2)

csf_na_grid = pi.aff_transform(csf, regis_aff, na_interp.shape, cval = csf.min()) 
t1_na_grid  = pi.aff_transform(t1, regis_aff, na_interp.shape, cval = t1.min()) 

# save the data

nib.save(nib.Nifti1Image(t1_na_grid, na_affine), './data/SodiumExample/T1_128_aligned.nii')
nib.save(nib.Nifti1Image(csf_na_grid, na_affine), './data/SodiumExample/csf_128_aligned.nii')
nib.save(nib.Nifti1Image(na_interp, na_affine), './data/SodiumExample/na_128.nii')

np.savetxt('./data/SodiumExample/affine_128.txt', regis_aff)



#return f, signal

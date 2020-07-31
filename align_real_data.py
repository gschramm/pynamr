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

sdir1 = os.path.join(glob(os.path.join(pdir,'*TE03'))[0], sdir.split('_')[0], sdir)
sdir2 = os.path.join(glob(os.path.join(pdir,'*TE5'))[0], sdir.split('_')[0], sdir)
fpattern = '*.c?'

t1_nii = nib.load(os.path.join(pdir,'mprage.nii'))
t1_nii = nib.as_closest_canonical(t1_nii)
t1     = t1_nii.get_fdata()
t1_affine = t1_nii.affine

#----
# load the complex coil images for first echo
data_shape = (64,64,64)

fnames = glob(os.path.join(sdir1, fpattern))
ncoils = len(fnames)
data   = np.zeros((ncoils,) + data_shape, dtype = np.complex64)

for i, fname in enumerate(fnames):
  data[i,...] = np.flip(np.fromfile(fname, dtype = np.complex64).reshape(data_shape).swapaxes(0,2), (0,2))
  
na = np.flip(np.abs(data).sum(axis = 0), (0,1))
na_norm = na.max()
na /= na_norm

#----
# load the complex coil images for second echo

fnames2 = glob(os.path.join(sdir2, fpattern))
ncoils2 = len(fnames2)
data2   = np.zeros((ncoils2,) + data_shape, dtype = np.complex64)

for i, fname2 in enumerate(fnames2):
  data2[i,...] = np.flip(np.fromfile(fname2, dtype = np.complex64).reshape(data_shape).swapaxes(0,2), (0,2))
  
na2 = np.flip(np.abs(data2).sum(axis = 0), (0,1))
na2 /= na_norm


#----------------------------------------------------------------------
# align the T1 to the sodium image

# interpolate sodium image to a 256 grid
na_interp  = zoom(na,  2, order = 1, prefilter = False)
na2_interp = zoom(na2, 2, order = 1, prefilter = False)

fov = 223.

na_voxsize = fov / np.array(na_interp.shape)

na_affine = np.diag(np.concatenate((na_voxsize,[1])))
na_affine[:-1,-1] = -fov/2.

# this is the affine that maps from the T1 onto the Na grid
pre_affine = np.linalg.inv(t1_affine) @ na_affine

#args = (na_interp, csf, True, True, pm.neg_mutual_information, pre_affine), 
reg_params = np.zeros(6)
res = minimize(pm.regis_cost_func, reg_params, 
               args = (na_interp, t1, True, True, pm.neg_mutual_information, pre_affine), 
               method = 'Powell', 
               options = {'ftol':1e-2, 'xtol':1e-2, 'disp':True, 'maxiter':20, 'maxfev':5000})

reg_params = res.x.copy()

regis_aff = pre_affine @ pi.kul_aff(reg_params, origin = np.array(na_interp.shape)/2)

#csf_na_grid = pi.aff_transform(csf, regis_aff, na_interp.shape, cval = csf.min()) 
t1_na_grid  = pi.aff_transform(t1, regis_aff, na_interp.shape, cval = t1.min()) 

# save the data

#nib.save(nib.Nifti1Image(csf_na_grid, na_affine), './data/SodiumExample/csf_128_aligned.nii')
nib.save(nib.Nifti1Image(t1_na_grid, na_affine), os.path.join(pdir,'mprage_128_aligned.nii'))
nib.save(nib.Nifti1Image(na_interp, na_affine), os.path.join(pdir,'TE03_128_' + sdir + '.nii'))
nib.save(nib.Nifti1Image(na2_interp, na_affine), os.path.join(pdir,'TE5_128_' + sdir + '.nii'))
 
np.savetxt(os.path.join(pdir,'affine_128.txt'), regis_aff)

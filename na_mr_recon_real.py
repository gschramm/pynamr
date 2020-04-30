import os
import numpy as np
import nibabel as nib

from glob import glob

def read_read_data(sdir = './data/SodiumExample/TBI-n005_tpiRecon_FID125462_TE03/PhyCha/kw1', fpattern = '*.c?'): 

  #t1_nii = nib.load('./data/SodiumExample/T1.nii')
  #t1_nii = nib.as_closest_canonical(t1_nii)
  #t1     = np.flip(t1_nii.get_fdata(), (0,1))
  
  #----
  # load the complex coil images
  
  fnames = glob(os.path.join(sdir, fpattern))
  
  ncoils = len(fnames)
  
  data_shape = (64,64,64)
  data = np.zeros((ncoils,) + data_shape, dtype = np.complex64)
  
  
  for i, fname in enumerate(fnames):
    data[i,...] = np.flip(np.fromfile(fname, dtype = np.complex64).reshape(data_shape).swapaxes(0,2), (0,2))
    
  s1 = np.abs(data).sum(axis = 0)
  s2 = np.abs(data.sum(axis = 0))
  
  s3 = np.flip(np.fromfile(os.path.join(sdir,'tpirec.csos'), 
                           dtype = np.complex64).reshape(data_shape).swapaxes(0,2), (0,2))
  
  f_3d   = s1.copy()
  f      = f_3d[:,:,26]
  signal = np.fft.fft2(f).astype(np.complex128)
  signal = signal.view('(2,)float')
  
  f = np.stack((f,np.zeros(f.shape)), axis = -1)

  return f, signal

import numpy as np
from   numba import njit, prange

#--------------------------------------------------------------
def apodized_fft_2d(f, readout_inds, apo_images):
  """ Calculate 2D apodized FFT of an image (e.g. caused by T2* decay during readout
  
  Parameters
  ----------

  f : 2d numpy array of shape (n0,n1)

  readout_inds : list of array indices (nr elements)
    containing the 2D array indicies that read out at every time point

  apo_images : 3d numpt array of shape(nr,n0n,1)
    containing the nultiplicative apodization images at each readout time point
  """

  F = np.zeros(f.shape, dtype = np.complex)

  for i in range(apo_images.shape[0]):
    tmp = np.fft.fft2(apo_images[i,...] * f)
    F[readout_inds[i]] = tmp[readout_inds[i]]

  # we normalize to get the norm of the operator to the norm of the gradient op
  F *= np.sqrt(4*f.ndim) / np.sqrt(np.prod(f.shape))

  return F

#--------------------------------------------------------------
def adjoint_apodized_fft_2d(F, readout_inds, apo_images):
  """ Calculate 2D apodized FFT of an image (e.g. caused by T2* decay during readout
  
  Parameters
  ----------

  F : 2d numpy array of shape (n0,n1)

  readout_inds : list of array indices (nr elements)
    containing the 2D array indicies that read out at every time point

  apo_images : 3d numpt array of shape(nr,n0n,1)
    containing the nultiplicative apodization images at each readout time point
  """

  n0, n1 = F.shape
  f      = np.zeros(F.shape, dtype = np.complex)

  for i in range(apo_images.shape[0]):
    tmp = np.zeros(f.shape, dtype = np.complex)
    tmp[readout_inds[i]] = F[readout_inds[i]]

    f += apo_images[i,...] * np.fft.ifft2(tmp)

  f *=  (np.sqrt(np.prod(F.shape)) * np.sqrt(4*F.ndim))

  return f

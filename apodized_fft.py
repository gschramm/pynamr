import numpy as np

#--------------------------------------------------------------
def apodized_fft(f, readout_inds, apo_imgs):
  """ Calculate apodized FFT of an image (e.g. caused by T2* decay during readout
  
  Parameters
  ----------

  f : a float64 numpy array of shape (n0,n1,...,nn,2)
    [...,0] is considered as the real part
    [...,1] is considered as the imag part

  readout_inds : list of array indices (nr elements)
    containing the 2D array indicies that read out at every time point

  apo_imgs : 3d numpy array of shape(nr,n0,n1,...,nn)
    containing the multiplicative apodization images at each readout time point

  Returns
  -------
  a float64 numpy array of shape (n0,n1,...,nn,2)
  """

  # create a complex view of the input real input array with two channels
  f = np.squeeze(f.view(dtype=np.complex128))

  F = np.zeros(f.shape, dtype = np.complex128)

  for i in range(apo_imgs.shape[0]):
    tmp = np.fft.fft2(apo_imgs[i,...] * f, axes = -np.arange(f.ndim,0,-1))
    F[readout_inds[i]] = tmp[readout_inds[i]]

  
  # we normalize to get the norm of the operator to the norm of the gradient op
  F *= np.sqrt(4*f.ndim) / np.sqrt(np.prod(f.shape))

  # convert F back to 2 real arrays
  f  = f.view('(2,)float')
  F  = F.view('(2,)float')

  return F

#--------------------------------------------------------------
def adjoint_apodized_fft(F, readout_inds, apo_imgs):
  """ Calculate apodized FFT of an image (e.g. caused by T2* decay during readout)
  
  Parameters
  ----------

  F : a float64 numpy array of shape (n0,n1,...,nn,2)
    [...,0] is considered as the real part
    [...,1] is considered as the imag part

  readout_inds : list of array indices (nr elements)
    containing the 2D array indicies that read out at every time point

  apo_imgs : 3d numpt array of shape(nr,n0,n1,...,nn)
    containing the nultiplicative apodization images at each readout time point

  Returns
  -------
  a float64 numpy array of shape (n0,n1,...,nn,2)
  """

  # create a complex view of the input real input array with two channels
  F = np.squeeze(F.view(dtype = np.complex128))

  f = np.zeros(F.shape, dtype = np.complex128)

  for i in range(apo_imgs.shape[0]):
    tmp = np.zeros(f.shape, dtype = np.complex128)
    tmp[readout_inds[i]] = F[readout_inds[i]]

    f += apo_imgs[i,...] * np.fft.ifft2(tmp, axes = -np.arange(F.ndim,0,-1))

  f *=  (np.sqrt(np.prod(F.shape)) * np.sqrt(4*F.ndim))

  # convert F back to 2 real arrays
  f  = f.view('(2,)float')
  F  = F.view('(2,)float')

  return f

#--------------------------------------------------------------
def apo_images(readout_times, T2star_short, T2star_long, C_short = 0.6, C_long = 0.4):
  apo_imgs = np.zeros((readout_times.shape[0],) + T2star_short.shape)

  for i, t_read in enumerate(readout_times):
    apo_imgs[i,...] = C_short*np.exp(-t_read / T2star_short) + C_long*np.exp(-t_read / T2star_long)

  return apo_imgs

#--------------------------------------------------------------
def apodized_fft_multi_echo(f, readout_inds, Gamma, t, dt, nechos = 2):
  """ Calculate apodized FFT of an image (e.g. caused by T2* decay during readout
  
  Parameters
  ----------

  f : a float64 numpy array of shape (n0,n1,...,nn,2)
    [...,0] is considered as the real part
    [...,1] is considered as the imag part

  readout_inds : list of array indices (nr elements)
    containing the 2D array indicies that read out at every time point

  Gamma : 3d numpy array of (n0,n1,...,nn)
    containing the exponential of the T2* time exp(-dt/T2*)

  t : float64 1d numpy array
    containing the readout times

  dt : float64
    the time difference between two adjacent echos

  nechos : int
    number of echos

  Returns
  -------
  a float64 numpy array of shape (nechos,n0,n1,2)
  """

  # create a complex view of the input real input array with two channels
  f  = np.squeeze(f.view(dtype=np.complex128))

  # signal of for all echos
  F = np.zeros((nechos,) + f.shape, dtype = np.complex128)

  for i in range(t.shape[0]):
    for k in range(nechos):
      tmp = np.fft.fftn((Gamma**((t[i]/dt)+k)) * f, axes = -np.arange(f.ndim,0,-1))
      F[k,...][readout_inds[i]] = tmp[readout_inds[i]]

  # we normalize to get the norm of the operator to the norm of the gradient op
  F *= np.sqrt(4*f.ndim) / np.sqrt(np.prod(f.shape))

  # convert F back to 2 real arrays
  f  = f.view('(2,)float')
  F  = F.view('(2,)float')

  return F

#--------------------------------------------------------------
def adjoint_apodized_fft_multi_echo(F, readout_inds, Gamma, t, dt, grad_gamma = False):
  """ Calculate apodized FFT of an image (e.g. caused by T2* decay during readout)

  Parameters
  ----------

  F : a float64 numpy array of shape (2, n0,n1,...,nn,2)
    [...,0] is considered as the real part
    [...,1] is considered as the imag part
    [0,...] is the signal for the first echo
    [1,...] is the signal for the second echo

  readout_inds : list of array indices (nr elements)
    containing the 2D array indicies that read out at every time point

  Gamma : 3d numpy array of (n0,n1,...,nn)
    containing the exponential of the T2* time exp(-dt/T2*)

  t : float64 1d numpy array
    containing the readout times

  dt : float64
    the time difference between the two adjacent echos

  grad_gamma : bool (default False)
    wether to calculate the adjoint needed for the gradient with respect to Gamma

  Returns
  -------
  a float64 numpy array of shape (n0,n1,...,nn,2)
  """

  # create a complex view of the input real input array with two channels
  F = np.squeeze(F.view(dtype = np.complex128))

  f = np.zeros(F[0,...].shape, dtype = np.complex128)

  nechos = F.shape[0]

  for i in range(t.shape[0]):
    for k in range(nechos):
      tmp = np.zeros(f.shape, dtype = np.complex128)
      tmp[readout_inds[i]] = F[k,...][readout_inds[i]]

      if grad_gamma:
        f += ((t[i]/dt) + k)*(Gamma**((t[i]/dt) + k - 1)) * np.fft.ifftn(tmp, axes = -np.arange(F[0,...].ndim,0,-1))
      else:
        f += (Gamma**((t[i]/dt) + k)) * np.fft.ifftn(tmp, axes = -np.arange(F[0,...].ndim,0,-1))

  f *=  ((np.sqrt(np.prod(F[0,...].shape))) * np.sqrt(4*F[0,...].ndim))

  # convert F back to 2 real arrays
  f  = f.view('(2,)float')
  F  = F.view('(2,)float')

  return f



#--------------------------------------------------------------
#--------------------------------------------------------------
#--------------------------------------------------------------
#--------------------------------------------------------------

# adjointness test
if __name__ == '__main__':
  np.random.seed(0)
  
  nechos = 4
  n      = 256

  x = np.random.rand(n,n,2)

  if nechos > 1:
    y = np.random.rand(nechos,n,n,2)
  else:
    y = np.random.rand(n,n,2)
  
  # setup the frequency array as used in numpy fft
  tmp    = np.fft.fftfreq(x.shape[0])
  k0, k1 = np.meshgrid(tmp, tmp, indexing = 'ij')
  abs_k  = np.sqrt(k0**2 + k1**2)
  
  # generate array of k-space readout times
  n_readout_bins     = 50
  readout_ind_array  = (abs_k * (n_readout_bins**2) / abs_k.max()) // n_readout_bins
  readout_times      = 100*abs_k[readout_ind_array == (n_readout_bins-1)].mean() * np.linspace(0,1,n_readout_bins)
  readout_inds       = []
  
  for i, t_read in enumerate(readout_times):
    readout_inds.append(np.where(readout_ind_array == i))
 
  if nechos > 1:
    Gam     = np.random.rand(n,n)
    tr      = np.arange(n_readout_bins)/2.
    delta_t = 5.

    x_fwd = apodized_fft_multi_echo(x, readout_inds, Gam, tr, delta_t, nechos = nechos)
    y_back = adjoint_apodized_fft_multi_echo(y, readout_inds, Gam, tr, delta_t)

  else:
    # generate the signal apodization images
    apo_imgs  = apo_images(readout_times, 8*np.random.rand(n,n), 30*np.random.rand(n,n))

    x_fwd  = apodized_fft(x, readout_inds, apo_imgs)
    y_back = adjoint_apodized_fft(y, readout_inds, apo_imgs) 

  print((x_fwd*y).sum(),(x*y_back).sum())

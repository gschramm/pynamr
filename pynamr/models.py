import cupy as xp

def downsample(x, ds, axis = 0):

  ds_shape       = list(x.shape)
  ds_shape[axis] = ds_shape[axis] // ds 
  ds_shape       = tuple(ds_shape)

  x_ds = xp.zeros(ds_shape, dtype = x.dtype)

  sl = slice(None,None,None)

  for i in range(ds_shape[axis]):
    sl1 = x.ndim*[sl]
    sl1[axis] = slice(ds*i,ds*(i+1),None)
    sl1 = tuple(sl1)

    sl2 = x.ndim*[sl]
    sl2[axis] = slice(i,i+1,None)
    sl2 = tuple(sl2)

    x_ds[sl2] = x[sl1].sum(axis = axis, keepdims = True)
  
  # normalize by sqrt(ds) to get operator norm 1
  return x_ds / xp.sqrt(ds)

def upsample(x_ds, ds, axis = 0):

  up_shape       = list(x_ds.shape)
  up_shape[axis] = up_shape[axis] * ds 
  up_shape       = tuple(up_shape)

  x = xp.zeros(up_shape, dtype = x_ds.dtype)

  sl = slice(None,None,None)

  for i in range(x_ds.shape[axis]):
    sl1 = x_ds.ndim*[sl]
    sl1[axis] = slice(ds*i,ds*(i+1),None)
    sl1 = tuple(sl1)

    sl2 = x_ds.ndim*[sl]
    sl2[axis] = slice(i,i+1,None)
    sl2 = tuple(sl2)

    x[sl1] = x_ds[sl2]

  # normalize by sqrt(ds) to get operator norm 1
  return x / xp.sqrt(ds)


#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

class DualTESodiumAcqModel:
  def __init__(self, image_shape, ncoils, sens, dt):
    # shape of the data
    self._data_shape = (64,64,64)
    # shape of the image
    self._image_shape = image_shape
    # downsampling factor
    self._ds = self._image_shape[0] // self._data_shape[0]

    # number coils
    self._ncoils = ncoils
    # sensitivity "image" for each coil in downsampled data space
    self._sens = sens
    # time between two echos
    self._dt = dt

    # parameters related to the read out trajectory
    self._eta            = 0.9830
    self._c1             = 0.54
    self._c2             = 0.46 
    self._alpha_sw_tpi   = 18.95
    self._beta_sw_tpi    = -0.5171
    self._t0_sw          = 0.0018
    self._k_edge         = 1.8*0.8197
    self._n_readout_bins = 32

    self.setup_readout()

  def setup_readout(self):
    print('setting up readout')

    # setup the readout time and readout indexes
    k0,k1,k2 = xp.meshgrid(xp.arange(64) - 32 + 0.5, xp.arange(64) - 32 + 0.5, xp.arange(64) - 32 + 0.5)
    abs_k = xp.sqrt(k0**2 + k1**2 + k2**2)
    abs_k = xp.fft.fftshift(abs_k)

    # rescale abs_k such that k = 1.8*0.8197 is at r = 32 (the edge)
    abs_k *= self._k_edge/31.5

    # calculate the readout times and the k-spaces locations that
    # are read at a given time
    t_read_3 = self.readout_time(abs_k, self._eta, self._c1, self._c2, self._alpha_sw_tpi,
                                 self._beta_sw_tpi, self._t0_sw)
  
    
    k_1d = xp.linspace(0, self._k_edge, self._n_readout_bins + 1)
    
    self._readout_inds = []
    self._tr = xp.zeros(self._n_readout_bins)
    
    for i in range(self._n_readout_bins):
      k_start = k_1d[i]
      k_end   = k_1d[i+1]
      rinds   = xp.where(xp.logical_and(abs_k >= k_start, abs_k <= k_end))
    
      self._tr[i] = t_read_3[rinds].mean()
      self._readout_inds.append(rinds)

  @property
  def image_shape(self):
    return self._image_shape 

  @image_shape.setter
  def image_shape(self, value):
    self._image_shape = value
    # update downsampling factor when image shape is changed
    self._ds = self._image_shape[0] // self._data_shape[0]

  @property
  def ncoils(self):
    return self._ncoils

  @ncoils.setter
  def ncoils(self, value):
    self._ncoils = value

  @property
  def sens(self):
    return self._sens

  @sens.setter
  def sens(self, value):
    self._sens = value

  @property
  def dt(self):
    return self._dt

  @dt.setter
  def dt(self, value):
    self._dt = value

  #------------------------------------------------------------------------------
  def forward(self, f, Gam):
    """ Calculate downsampled FFT of an image f
     ha
        Parameters
        ----------
        
        f : a float64 numpy/cupy array of shape (n0,n1,n2,2)
          [...,0] is considered as the real part
          [...,1] is considered as the imag part
        
        
        Returns
        -------
        a float64 numpy/cupy array of shape (n0,n1,n2,2)
    """

    # create a complex view of the input real input array with two channels
    f    = xp.squeeze(f.view(dtype = xp.complex64), axis = -1)

    # downsample f
    f_ds = downsample(downsample(downsample(f, self._ds, axis = 0), self._ds, axis = 1), self._ds, axis = 2)
    Gam_ds = downsample(downsample(downsample(Gam, self._ds, axis = 0), self._ds, axis = 1), self._ds, axis = 2)

    F = xp.zeros((ncoils,2,) + f_ds.shape, dtype = xp.complex64)

    for i_sens in range(self._ncoils):
      F[i_sens,0,...] = xp.fft.fftn(self.sens[i_sens,...] * f_ds, norm = 'ortho')
      F[i_sens,1,...] = xp.fft.fftn(self.sens[i_sens,...] * Gam_ds*f_ds, norm = 'ortho')

    # convert complex64 arrays back to 2 float32 array
    f = xp.stack([f.real, f.imag], axis = -1)
    F = xp.stack([F.real, F.imag], axis = -1)

    return F

  #------------------------------------------------------------------------------
  def adjoint(self, F, Gam):
    """ Calculate the adjoint of the downsampled FFT of a k-space image F
      
        Parameters
        ----------
        
        F : a float64 numpy/cupy array of shape (n0,n1,n2,2)
          [...,0] is considered as the real part
          [...,1] is considered as the imag part
        
        
        Returns
        -------
        a float64 numpy/cupy array of shape (n0,n1,n2,2)
    """

    # create a complex view of the input real input array with two channels
    F  = xp.squeeze(F.view(dtype = xp.complex64), axis = -1)

    f_ds = xp.zeros(F.shape[2:], dtype = xp.complex64)

    Gam_ds = downsample(downsample(downsample(Gam, self._ds, axis = 0), self._ds, axis = 1), self._ds, axis = 2)

    for i_sens in range(self._ncoils):
      f_ds += xp.conj(self.sens[i_sens])*xp.fft.ifftn(F[i_sens,0,...], norm = 'ortho')
      f_ds += Gam_ds*xp.conj(self.sens[i_sens])*xp.fft.ifftn(F[i_sens,1,...], norm = 'ortho')

    # upsample f
    f = upsample(upsample(upsample(f_ds, self._ds, axis = 0), self._ds, axis = 1), self._ds, axis = 2)

    # convert complex64 arrays back to 2 float32 array
    f = xp.stack([f.real, f.imag], axis = -1)
    F = xp.stack([F.real, F.imag], axis = -1)

    return f

  #------------------------------------------------------------------------------
  @staticmethod
  def readout_time(k, eta, c1, c2, alpha_sw_tpi, beta_sw_tpi, t0_sw):
  
    # the point until the readout is linear
    m = 1126 * 0.16
  
    k_lin = t0_sw * m
  
    i1 = xp.where(k <= k_lin)
    i2 = xp.where(k > k_lin)
  
    t = xp.zeros(k.shape)
    t[i1] = k[i1] / m
    t[i2] = t0_sw + ((c2*(k[i2]**3) - ((c1/eta)*xp.exp(-eta*(k[i2]**3))) - beta_sw_tpi) / (3*alpha_sw_tpi))

    # convert to ms
    t *= 1000

    return t


#---------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

if __name__ == '__main__':

  ds = 2
  n  = 128 
  ncoils = 8
  dt = 5.

  a = xp.pad(xp.random.rand(n-4,n-4,n-4),2).astype(xp.float32)

  f = xp.stack([a,a], axis = -1)

  sens = xp.random.rand(ncoils,64,64,64).astype(xp.float32) + 1j*xp.random.rand(ncoils,64,64,64).astype(xp.float32)
 
  Gam = xp.random.rand(n,n,n).astype(xp.float32)
  
  m = DualTESodiumAcqModel(a.shape, ncoils, sens, dt)
  
  f_fwd  = m.forward(f, Gam)
  F      = xp.random.rand(*f_fwd.shape).astype(xp.float32)
  F_back = m.adjoint(F, Gam) 
  print((f_fwd*F).sum() / (f*F_back).sum())

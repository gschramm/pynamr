import numpy as np

#TODO: - grad_x = adjoint(outer)
#      - grad_y = inner_gamma(outer)
#                 "grad_gamma" is similar to adjoint (with different exponent for gamm

def downsample(x, ds, xp, axis = 0):

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
  return x_ds / np.sqrt(ds)

def upsample(x_ds, ds, xp, axis = 0):

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
  return x / np.sqrt(ds)


#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

class MonoExpDualTESodiumAcqModel:
  """ mono exponential decay model for dual TE Na MR data

      Parameters
      ----------

      image_shape ... tuple
        shape of the image

      ncoils ... int
        number of coils

      sens ... 3D complex128 numpy arrays of shape (ncoils, image_shape) 
        including the coil sensitivities

      dt ... float
        time difference between the first and second echo in ms
        the first echo is assumed to be at 0ms
  """

  def __init__(self, data_shape, ds, ncoils, sens, dt, xp):
    # shape of the data
    self._data_shape = data_shape
    # downsampling factor
    self._ds = ds
    # shape of the image
    self._image_shape = tuple([ds*x for x in self._data_shape])

    # number coils
    self._ncoils = ncoils
    # sensitivity "image" for each coil in downsampled data space
    self._sens = sens
    # time between two echos
    self._dt = dt

    # numpy / cupy module to use for ffts
    self._xp = xp

    # send sens to GPU if needed
    if self._xp.__name__ == 'cupy':
      self._sens = self._xp.asarray(self._sens)

    # parameters related to the read out trajectory
    self._eta            = 0.9830
    self._c1             = 0.54
    self._c2             = 0.46 
    self._alpha_sw_tpi   = 18.95
    self._beta_sw_tpi    = -0.5171
    self._t0_sw          = 0.0018
    self._k_edge         = 1.8*0.8197
    self._n_readout_bins = self._data_shape[0] // 2

    self.setup_readout()

  def setup_readout(self):
    # setup the readout time and readout indexes
    k0,k1,k2 = np.meshgrid(np.linspace(-self._k_edge, self._k_edge, self._data_shape[0]),
                           np.linspace(-self._k_edge, self._k_edge, self._data_shape[1]),
                           np.linspace(-self._k_edge, self._k_edge, self._data_shape[2]))

    abs_k = np.sqrt(k0**2 + k1**2 + k2**2)
    abs_k = np.fft.fftshift(abs_k)


    # calculate the readout times and the k-spaces locations that
    # are read at a given time
    t_read_3 = self.readout_time(abs_k, self._eta, self._c1, self._c2, self._alpha_sw_tpi,
                                 self._beta_sw_tpi, self._t0_sw)

    k_1d = np.linspace(0, self._k_edge, self._n_readout_bins + 1)
    
    self._readout_inds = []
    self._tr = np.zeros(self._n_readout_bins, dtype = np.float64)

    self._kmask = np.zeros(self._data_shape, dtype = np.uint8)
    
    for i in range(self._n_readout_bins):
      k_start = k_1d[i]
      k_end   = k_1d[i+1]
      rinds   = np.where(np.logical_and(abs_k >= k_start, abs_k <= k_end))
    
      self._tr[i] = t_read_3[rinds].mean()
      self._readout_inds.append(rinds)
      self._kmask[rinds] = 1

    # convert kmask mask into a pseudo "complex" array
    self._kmask = np.stack([self._kmask, self._kmask], axis = -1)

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
    # send sens to GPU if needed
    if self._xp.__name__ == 'cupy':
      self._sens = self._xp.asarray(self._sens)

  @property
  def dt(self):
    return self._dt

  @dt.setter
  def dt(self, value):
    self._dt = value

  @property
  def n_readout_bins(self):
    return self._n_readout_bins

  @n_readout_bins.setter
  def n_readout_bins(self, value):
    self._n_readout_bins = value
    self.setup_readout()

  @property
  def tr(self):
    return self._tr

  @property
  def eta(self):
    return self._eta

  @eta.setter
  def eta(self, value):
    self._eta = value
    self.setup_readout()

  @property
  def c1(self):
    return self._c1

  @c1.setter
  def c1(self, value):
    self._c1 = value
    self.setup_readout()

  @property
  def c2(self):
    return self._c2

  @c2.setter
  def c2(self, value):
    self._c2 = value
    self.setup_readout()

  @property
  def alpha_sw_tpi(self):
    return self._alpha_sw_tpi

  @alpha_sw_tpi.setter
  def alpha_sw_tpi(self, value):
    self._alpha_sw_tpi = value
    self.setup_readout()

  @property
  def beta_sw_tpi(self):
    return self._beta_sw_tpi

  @beta_sw_tpi.setter
  def beta_sw_tpi(self, value):
    self._beta_sw_tpi = value
    self.setup_readout()

  @property
  def t0_sw(self):
    return self._t0_sw

  @t0_sw.setter
  def t0_sw(self, value):
    self._t0_sw = value
    self.setup_readout()

  @property
  def k_edge(self):
    return self._k_edge

  @k_edge.setter
  def k_edge(self, value):
    self._k_edge = value
    self.setup_readout()

  @property
  def kmask(self):
    return self._kmask

  #------------------------------------------------------------------------------
  def forward(self, f, Gam):
    """ Calculate downsampled FFT of an image f

        Parameters
        ----------
        
        f : a float64 numpy/cupy array of shape (self.image_shape,2)
          [...,0] is considered as the real part
          [...,1] is considered as the imag part

        Gam : a float64 numpy/cupy array of shape (self.image_shape)
        
        Returns
        -------
        a float64 numpy/cupy array of shape (self.ncoils,self.image_shape,2)
    """

    # create a complex view of the input real input array with two channels
    f = np.squeeze(f.view(dtype = np.complex128), axis = -1)

    #----------------------
    # send f and Gam to GPU
    if self._xp.__name__ == 'cupy':
      f   = self._xp.asarray(f)
      Gam = self._xp.asarray(Gam)

    # downsample f and Gamma
    f_ds = downsample(downsample(downsample(f, self._ds, self._xp, axis = 0), self._ds, self._xp, axis = 1),                                            self._ds, self._xp, axis = 2)
    Gam_ds = downsample(downsample(downsample(Gam, self._ds, self._xp, axis = 0), 
                                              self._ds, self._xp, axis = 1), self._ds, self._xp, axis = 2)

    F = self._xp.zeros((self._ncoils,2,) + f_ds.shape, dtype = self._xp.complex128)

    for i_sens in range(self._ncoils):
      for it in range(self._n_readout_bins):
        F[i_sens,0,...][self._readout_inds[it]] = self._xp.fft.fftn(self.sens[i_sens,...] * Gam_ds**(self._tr[it]/self._dt) * f_ds, norm = 'ortho')[self._readout_inds[it]]
        F[i_sens,1,...][self._readout_inds[it]] = self._xp.fft.fftn(self.sens[i_sens,...] * Gam_ds**((self._tr[it]/self._dt) + 1) * f_ds, norm = 'ortho')[self._readout_inds[it]]

    # get f, F, Gam back from GPU
    if self._xp.__name__ == 'cupy':
      F   = self._xp.asnumpy(F)
    #----------------------

    # convert complex128 arrays back to 2 float64 array
    F = np.stack([F.real, F.imag], axis = -1)

    return F

  #------------------------------------------------------------------------------
  def adjoint(self, F, Gam):
    """ Calculate the adjoint of the downsampled FFT of a k-space image F
      
        Parameters
        ----------
        
        F : a float64 numpy/cupy array of shape (self.ncoils,self.image_shape,2)
          [...,0] is considered as the real part
          [...,1] is considered as the imag part
        
        Gam : a float64 numpy/cupy array of shape (self.image_shape)
        
        Returns
        -------
        a float64 numpy/cupy array of shape (self.image_shape)
    """

    # create a complex view of the input real input array with two channels
    F  = np.squeeze(F.astype(np.float64).view(dtype = np.complex128), axis = -1)

    #----------------------
    # send F, Gam to GPU
    if self._xp.__name__ == 'cupy':
      F   = self._xp.asarray(F)
      Gam = self._xp.asarray(Gam)

    f_ds = self._xp.zeros(F.shape[2:], dtype = self._xp.complex128)

    Gam_ds = downsample(downsample(downsample(Gam, self._ds, self._xp, axis = 0), 
                        self._ds, self._xp, axis = 1), self._ds, self._xp, axis = 2)

    for i_sens in range(self._ncoils):
      for it in range(self._n_readout_bins):
        tmp0 = self._xp.zeros(F[i_sens,0,...].shape, dtype = F.dtype)
        tmp0[self._readout_inds[it]] = F[i_sens,0,...][self._readout_inds[it]]
        f_ds += (Gam_ds**(self._tr[it]/self._dt)) * self._xp.conj(self.sens[i_sens])*self._xp.fft.ifftn(tmp0, norm = 'ortho')

        tmp1 = self._xp.zeros(F[i_sens,1,...].shape, dtype = F.dtype)
        tmp1[self._readout_inds[it]] = F[i_sens,1,...][self._readout_inds[it]]
        f_ds += (Gam_ds**((self._tr[it]/self._dt) + 1)) * self._xp.conj(self.sens[i_sens])*self._xp.fft.ifftn(tmp1, norm = 'ortho')

    # upsample f
    f = upsample(upsample(upsample(f_ds, self._ds, self._xp, axis = 0), self._ds, self._xp, axis = 1), 
                                   self._ds, self._xp, axis = 2)

    # get f, F, Gam back from GPU
    if self._xp.__name__ == 'cupy':
      f   = self._xp.asnumpy(f)
    #----------------------

    # convert complex128 arrays back to 2 float64 array
    f = np.stack([f.real, f.imag], axis = -1)

    return f

  #------------------------------------------------------------------------------

  def grad_gam(self, F, Gam, img):
    """ Calculate the the "inner" derivative with respect to Gamma
      
        Parameters
        ----------
        
        F : a float64 numpy/cupy array of shape (self.ncoils,self.image_shape,2)
            containing the "outer derivative" of the cost function
          [...,0] is considered as the real part
          [...,1] is considered as the imag part
        
        Gam : a float64 numpy/cupy array of shape (self.image_shape)
        
        img : a float64 numpy/cupy array of shape (self.image_shape,2)
            containing the current Na concentration image
          [...,0] is considered as the real part
          [...,1] is considered as the imag part

        Returns
        -------
        a float64 numpy/cupy array of shape (self.image_shape)
    """

    # create a complex view of the input real input array with two channels
    F  = np.squeeze(F.astype(np.float64).view(dtype = np.complex128), axis = -1)

    img = np.squeeze(img.view(dtype = np.complex128), axis = -1)

    #----------------------
    # send F, Gam to GPU
    if self._xp.__name__ == 'cupy':
      F   = self._xp.asarray(F)
      Gam = self._xp.asarray(Gam)
      img = self._xp.asarray(img)

    f_ds = self._xp.zeros(F.shape[2:], dtype = self._xp.complex128)

    Gam_ds = downsample(downsample(downsample(Gam, self._ds, self._xp, axis = 0), 
                        self._ds, self._xp, axis = 1), self._ds, self._xp, axis = 2)

    img_ds = downsample(downsample(downsample(img, self._ds, self._xp, axis = 0), 
                        self._ds, self._xp, axis = 1), self._ds, self._xp, axis = 2)


    for i_sens in range(self._ncoils):
      for it in range(self._n_readout_bins):
        n =  self._tr[it]/self._dt

        tmp0 = self._xp.zeros(F[i_sens,0,...].shape, dtype = F.dtype)
        tmp0[self._readout_inds[it]] = F[i_sens,0,...][self._readout_inds[it]]
        f_ds += n * (Gam_ds**(n - 1)) * self._xp.conj(img_ds * self.sens[i_sens])*self._xp.fft.ifftn(tmp0, norm = 'ortho')

        tmp1 = self._xp.zeros(F[i_sens,1,...].shape, dtype = F.dtype)
        tmp1[self._readout_inds[it]] = F[i_sens,1,...][self._readout_inds[it]]
        f_ds += (n + 1) * (Gam_ds**n) * self._xp.conj(img_ds * self.sens[i_sens])*self._xp.fft.ifftn(tmp1, norm = 'ortho')

    # upsample f
    f = upsample(upsample(upsample(f_ds, self._ds, self._xp, axis = 0), self._ds, self._xp, axis = 1), 
                                   self._ds, self._xp, axis = 2)

    # get f, F, Gam back from GPU
    if self._xp.__name__ == 'cupy':
      f   = self._xp.asnumpy(f)
    #----------------------

    # convert complex128 arrays back to 2 float64 array
    f = np.stack([f.real, f.imag], axis = -1)

    return f[...,0]


  #------------------------------------------------------------------------------
  @staticmethod
  def readout_time(k, eta, c1, c2, alpha_sw_tpi, beta_sw_tpi, t0_sw):
    # the point until the readout is linear
    m = 1126 * 0.16
  
    k_lin = t0_sw * m
  
    i1 = np.where(k <= k_lin)
    i2 = np.where(k > k_lin)
  
    t = np.zeros(k.shape)
    t[i1] = k[i1] / m
    t[i2] = t0_sw + ((c2*(k[i2]**3) - ((c1/eta)*np.exp(-eta*(k[i2]**3))) - beta_sw_tpi) / (3*alpha_sw_tpi))

    # convert to ms
    t *= 1000

    return t


#---------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

if __name__ == '__main__':

  import cupy as cp

  np.random.seed(1)

  data_shape = (64,64,64)
  ds = 2
  ncoils = 4
  dt = 5.
  xp = cp

  n_ds = data_shape[0] 
  n    = ds*n_ds


  a = np.pad(np.random.rand(n-4,n-4,n-4),2).astype(np.float64)
  f = np.stack([a,a], axis = -1)

  sens = np.random.rand(ncoils,n_ds,n_ds,n_ds).astype(np.float64) + 1j*np.random.rand(ncoils,n_ds,n_ds,n_ds).astype(np.float64)
 
  Gam = np.random.rand(n,n,n).astype(np.float64)
  
  m = MonoExpDualTESodiumAcqModel(data_shape, ds, ncoils, sens, dt, xp)
  
  f_fwd  = m.forward(f, Gam)
  F      = np.random.rand(*f_fwd.shape).astype(np.float64)
  F_back = m.adjoint(F, Gam) 
  print((f_fwd*F).sum() / (f*F_back).sum())

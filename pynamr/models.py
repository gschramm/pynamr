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

    # downsample f and Gamma
    f_ds = downsample(downsample(downsample(f, self._ds, axis = 0), self._ds, axis = 1), self._ds, axis = 2)
    Gam_ds = downsample(downsample(downsample(Gam, self._ds, axis = 0), self._ds, axis = 1), self._ds, axis = 2)

    F = xp.zeros((ncoils,2,) + f_ds.shape, dtype = xp.complex64)

    for i_sens in range(self._ncoils):
      for it in range(self._n_readout_bins):
        F[i_sens,0,...][self._readout_inds[it]] = xp.fft.fftn(self.sens[i_sens,...] * Gam_ds**(self._tr[it]/self._dt) * f_ds, norm = 'ortho')[self._readout_inds[it]]
        F[i_sens,1,...][self._readout_inds[it]] = xp.fft.fftn(self.sens[i_sens,...] * Gam_ds**((self._tr[it]/self._dt) + 1) * f_ds, norm = 'ortho')[self._readout_inds[it]]

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
      for it in range(self._n_readout_bins):
        tmp0 = xp.zeros(F[i_sens,0,...].shape, dtype = F.dtype)
        tmp0[self._readout_inds[it]] = F[i_sens,0,...][self._readout_inds[it]]
        f_ds += (Gam_ds**(self._tr[it]/self._dt)) * xp.conj(self.sens[i_sens])*xp.fft.ifftn(tmp0, norm = 'ortho')

        tmp1 = xp.zeros(F[i_sens,1,...].shape, dtype = F.dtype)
        tmp1[self._readout_inds[it]] = F[i_sens,1,...][self._readout_inds[it]]
        f_ds += (Gam_ds**((self._tr[it]/self._dt) + 1)) * xp.conj(self.sens[i_sens])*xp.fft.ifftn(tmp1, norm = 'ortho')


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

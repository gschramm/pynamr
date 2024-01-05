import numpy as np
import cupy as cp


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

    F = np.zeros(f.shape, dtype=np.complex128)

    for i in range(apo_imgs.shape[0]):
        tmp = np.fft.fft2(apo_imgs[i, ...] * f, axes=-np.arange(f.ndim, 0, -1))
        F[readout_inds[i]] = tmp[readout_inds[i]]

    # we normalize to get the norm of the operator to the norm of the gradient op
    F *= np.sqrt(4 * f.ndim) / np.sqrt(np.prod(f.shape))

    # convert F back to 2 real arrays
    f = f.view('(2,)float')
    F = F.view('(2,)float')

    return F


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
    F = np.squeeze(F.view(dtype=np.complex128))

    f = np.zeros(F.shape, dtype=np.complex128)

    for i in range(apo_imgs.shape[0]):
        tmp = np.zeros(f.shape, dtype=np.complex128)
        tmp[readout_inds[i]] = F[readout_inds[i]]

        f += apo_imgs[i, ...] * np.fft.ifft2(tmp,
                                             axes=-np.arange(F.ndim, 0, -1))

    f *= (np.sqrt(np.prod(F.shape)) * np.sqrt(4 * F.ndim))

    # convert F back to 2 real arrays
    f = f.view('(2,)float')
    F = F.view('(2,)float')

    return f


def apo_images(readout_times,
               T2star_short,
               T2star_long,
               C_short=0.6,
               C_long=0.4):
    apo_imgs = np.zeros((readout_times.shape[0], ) + T2star_short.shape)

    for i, t_read in enumerate(readout_times):
        apo_imgs[i, ...] = C_short * np.exp(
            -t_read / T2star_short) + C_long * np.exp(-t_read / T2star_long)

    return apo_imgs


def apodized_fft_multi_echo(f, readout_inds, Gamma, t, dt, sens, nechos=2):
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

    # get numpy / cupy module depending on input
    xp = cp.get_array_module(f)

    # create a complex view of the input real input array with two channels
    f = xp.squeeze(f.view(dtype=xp.complex128), axis=-1)

    sens = xp.squeeze(sens.view(dtype=xp.complex128), axis=-1)
    ncoils = sens.shape[0]

    # signal of for all echos
    F = xp.zeros((
        ncoils,
        nechos,
    ) + f.shape, dtype=xp.complex128)

    for isens in range(ncoils):
        f_sens = f * sens[isens, ...]

        for i in range(t.shape[0]):
            for k in range(nechos):
                tmp = xp.fft.fftn((Gamma**((t[i] / dt) + k)) * f_sens,
                                  norm='ortho')
                F[isens, k, ...][readout_inds[i]] = tmp[readout_inds[i]]

    # convert F back to 2 real arrays
    f = xp.stack([f.real, f.imag], axis=-1)
    sens = xp.stack([sens.real, sens.imag], axis=-1)
    F = xp.stack([F.real, F.imag], axis=-1)

    return F


def adjoint_apodized_fft_multi_echo(F,
                                    readout_inds,
                                    Gamma,
                                    t,
                                    dt,
                                    sens,
                                    grad_gamma=False):
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

    # get numpy / cupy module depending on input
    xp = cp.get_array_module(F)

    # create a complex view of the input real input array with two channels
    F = xp.squeeze(F.view(dtype=xp.complex128), axis=-1)
    f = xp.zeros(F[0, 0, ...].shape, dtype=xp.complex128)

    sens = xp.squeeze(sens.view(dtype=xp.complex128), axis=-1)

    nechos = F.shape[1]
    ncoils = sens.shape[0]

    for isens in range(ncoils):
        for i in range(t.shape[0]):
            for k in range(nechos):
                tmp = xp.zeros(f.shape, dtype=xp.complex128)
                tmp[readout_inds[i]] = F[isens, k, ...][readout_inds[i]]

                if grad_gamma:
                    f += ((t[i] / dt) + k) * (Gamma**(
                        (t[i] / dt) + k - 1)) * xp.fft.ifftn(
                            tmp, norm='ortho') * xp.conj(sens[isens, ...])
                else:
                    f += (Gamma**((t[i] / dt) + k)) * xp.fft.ifftn(
                        tmp, norm='ortho') * xp.conj(sens[isens, ...])

    # convert F back to 2 real arrays
    f = xp.stack([f.real, f.imag], axis=-1)
    sens = xp.stack([sens.real, sens.imag], axis=-1)
    F = xp.stack([F.real, F.imag], axis=-1)

    return f


#--------------------------------------------------------------
#--------------------------------------------------------------
#--------------------------------------------------------------
#--------------------------------------------------------------

# adjointness test
if __name__ == '__main__':
    np.random.seed(0)

    nechos = 2
    ncoils = 4
    n = 128

    x = np.random.rand(n, n, n, 2)
    sens = np.random.rand(ncoils, n, n, n, 2)

    y = np.random.rand(ncoils, nechos, n, n, n, 2)

    # setup the frequency array as used in numpy fft
    tmp = np.fft.fftfreq(n)
    k0, k1, k2 = np.meshgrid(tmp, tmp, tmp, indexing='ij')
    abs_k = np.sqrt(k0**2 + k1**2 + k2**2)

    # generate array of k-space readout times
    n_readout_bins = 32
    readout_ind_array = (abs_k *
                         (n_readout_bins**2) / abs_k.max()) // n_readout_bins
    readout_times = 100 * abs_k[readout_ind_array == (
        n_readout_bins - 1)].mean() * np.linspace(0, 1, n_readout_bins)
    readout_inds = []

    for i, t_read in enumerate(readout_times):
        readout_inds.append(np.where(readout_ind_array == i))

    Gam = np.random.rand(n, n, n)
    tr = np.arange(n_readout_bins) / 2.
    delta_t = 5.

    x_fwd = apodized_fft_multi_echo(cp.asarray(x),
                                    readout_inds,
                                    cp.asarray(Gam),
                                    tr,
                                    delta_t,
                                    cp.asarray(sens),
                                    nechos=nechos).get()
    y_back = adjoint_apodized_fft_multi_echo(cp.asarray(y), readout_inds,
                                             cp.asarray(Gam), tr, delta_t,
                                             cp.asarray(sens)).get()

    print((x_fwd * y).sum(), (x * y_back).sum())

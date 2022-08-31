"""signal models for Na MR reconstruction"""
import typing
import numpy as np

# check whether cupy is available
try:
    import cupy as cp
except ModuleNotFoundError:
    import numpy as cp

from .utils import RadialKSpacePartitioner, XpArray
from .utils import complex_view_of_real_array, real_view_of_complex_array
from .utils import downsample, upsample

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------


class MonoExpDualTESodiumAcqModel:
    """ mono exponential decay model for dual TE Na MR data

        Parameters
        ----------

        ds ... downsample factor (image shape / data shape)

        sens ... complex array with coil sensitivities with shape (ncoils, data_shape)

        dt ... difference time between the two echos (ms)

        readout_time ... Callable that maps 1d kspace array to readout time [ms]
                         for used read out

        kspace_part ... RadialKSpacePartioner that partitions cartesian k-space volume
                        into radial shells of "same" readout time
    """

    def __init__(self, ds: int, sens: XpArray, dt: float,
                 readout_time: typing.Callable[[np.ndarray], np.ndarray],
                 kspace_part: RadialKSpacePartitioner) -> None:

        # downsampling factor
        self._ds = ds
        # sensitivity "image" for each coil in downsampled data space
        self._sens = sens
        # number coils
        self._ncoils = self._sens.shape[0]

        # time between two echos
        self._dt = dt

        # numpy / cupy module to use for ffts
        if isinstance(self._sens, np.ndarray):
            self._xp = np
        else:
            self._xp = cp

        # callable that calculates readout time from k array
        self._readout_time = readout_time

        # object that partions cartesian k-space into radial shells
        self._kspace_part = kspace_part
        # shape of the data
        self._data_shape = self._kspace_part._data_shape
        # shape of the image
        self._image_shape = tuple([ds * x for x in self._data_shape])

        # readout time of every shell
        self._tr = self._readout_time(self._kspace_part.k)

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return self._image_shape

    @property
    def ncoils(self) -> int:
        return self._ncoils

    @property
    def sens(self) -> XpArray:
        return self._sens

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def n_readout_bins(self) -> int:
        return self._kspace_part.n_readout_bins

    @property
    def readout_inds(self) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return self._kspace_part.readout_inds

    @property
    def tr(self) -> float:
        return self._tr

    @property
    def k_edge(self) -> float:
        return self._kspace_part.k_edge

    @property
    def kmask(self) -> np.ndarray:
        return self._kspace_part.kmask

    #------------------------------------------------------------------------------
    def forward(self, f: np.ndarray, Gam: np.ndarray) -> np.ndarray:
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
        f = complex_view_of_real_array(f)

        #----------------------
        # send f and Gam to GPU
        if self._xp.__name__ == 'cupy':
            f = self._xp.asarray(f)
            Gam = self._xp.asarray(Gam)

        # downsample f and Gamma
        f_ds = downsample(downsample(downsample(f, self._ds, axis=0),
                                     self._ds,
                                     axis=1),
                          self._ds,
                          axis=2)
        Gam_ds = downsample(downsample(downsample(Gam, self._ds, axis=0),
                                       self._ds,
                                       axis=1),
                            self._ds,
                            axis=2)

        F = self._xp.zeros((
            self._ncoils,
            2,
        ) + f_ds.shape,
                           dtype=self._xp.complex128)

        for i_sens in range(self._ncoils):
            for it in range(self.n_readout_bins):
                F[i_sens, 0, ...][self.readout_inds[it]] = self._xp.fft.fftn(
                    self.sens[i_sens, ...] *
                    Gam_ds**(self._tr[it] / self._dt) * f_ds,
                    norm='ortho')[self.readout_inds[it]]
                F[i_sens, 1, ...][self.readout_inds[it]] = self._xp.fft.fftn(
                    self.sens[i_sens, ...] *
                    Gam_ds**((self._tr[it] / self._dt) + 1) * f_ds,
                    norm='ortho')[self.readout_inds[it]]

        # get f, F, Gam back from GPU
        if self._xp.__name__ == 'cupy':
            F = self._xp.asnumpy(F)
        #----------------------

        # convert complex128 arrays back to 2 float64 array
        F = real_view_of_complex_array(F)

        return F

    #------------------------------------------------------------------------------
    def adjoint(self, F: np.ndarray, Gam: np.ndarray) -> np.ndarray:
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
        F = complex_view_of_real_array(F)

        #----------------------
        # send F, Gam to GPU
        if self._xp.__name__ == 'cupy':
            F = self._xp.asarray(F)
            Gam = self._xp.asarray(Gam)

        f_ds = self._xp.zeros(F.shape[2:], dtype=self._xp.complex128)

        Gam_ds = downsample(downsample(downsample(Gam, self._ds, axis=0),
                                       self._ds,
                                       axis=1),
                            self._ds,
                            axis=2)

        for i_sens in range(self._ncoils):
            for it in range(self.n_readout_bins):
                tmp0 = self._xp.zeros(F[i_sens, 0, ...].shape, dtype=F.dtype)
                tmp0[self.readout_inds[it]] = F[i_sens, 0,
                                                ...][self.readout_inds[it]]
                f_ds += (Gam_ds**(self._tr[it] / self._dt)) * self._xp.conj(
                    self.sens[i_sens]) * self._xp.fft.ifftn(tmp0, norm='ortho')

                tmp1 = self._xp.zeros(F[i_sens, 1, ...].shape, dtype=F.dtype)
                tmp1[self.readout_inds[it]] = F[i_sens, 1,
                                                ...][self.readout_inds[it]]
                f_ds += (Gam_ds**(
                    (self._tr[it] / self._dt) + 1)) * self._xp.conj(
                        self.sens[i_sens]) * self._xp.fft.ifftn(tmp1,
                                                                norm='ortho')

        # upsample f
        f = upsample(upsample(upsample(f_ds, self._ds, axis=0),
                              self._ds,
                              axis=1),
                     self._ds,
                     axis=2)

        # get f, F, Gam back from GPU
        if self._xp.__name__ == 'cupy':
            f = self._xp.asnumpy(f)
        #----------------------

        # convert complex128 arrays back to 2 float64 array
        f = real_view_of_complex_array(f)

        return f

    #------------------------------------------------------------------------------

    def grad_gam(self, F: np.ndarray, Gam: np.ndarray,
                 img: np.ndarray) -> np.ndarray:
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
        F = complex_view_of_real_array(F)
        img = complex_view_of_real_array(img)

        #----------------------
        # send F, Gam to GPU
        if self._xp.__name__ == 'cupy':
            F = self._xp.asarray(F)
            Gam = self._xp.asarray(Gam)
            img = self._xp.asarray(img)

        f_ds = self._xp.zeros(F.shape[2:], dtype=self._xp.complex128)

        Gam_ds = downsample(downsample(downsample(Gam, self._ds, axis=0),
                                       self._ds,
                                       axis=1),
                            self._ds,
                            axis=2)

        img_ds = downsample(downsample(downsample(img, self._ds, axis=0),
                                       self._ds,
                                       axis=1),
                            self._ds,
                            axis=2)

        for i_sens in range(self._ncoils):
            for it in range(self.n_readout_bins):
                n = self._tr[it] / self._dt

                tmp0 = self._xp.zeros(F[i_sens, 0, ...].shape, dtype=F.dtype)
                tmp0[self.readout_inds[it]] = F[i_sens, 0,
                                                ...][self.readout_inds[it]]
                f_ds += n * (Gam_ds**(n - 1)) * self._xp.conj(
                    img_ds * self.sens[i_sens]) * self._xp.fft.ifftn(
                        tmp0, norm='ortho')

                tmp1 = self._xp.zeros(F[i_sens, 1, ...].shape, dtype=F.dtype)
                tmp1[self.readout_inds[it]] = F[i_sens, 1,
                                                ...][self.readout_inds[it]]
                f_ds += (n + 1) * (Gam_ds**n) * self._xp.conj(
                    img_ds * self.sens[i_sens]) * self._xp.fft.ifftn(
                        tmp1, norm='ortho')

        # upsample f
        f = upsample(upsample(upsample(f_ds, self._ds, axis=0),
                              self._ds,
                              axis=1),
                     self._ds,
                     axis=2)

        # get f, F, Gam back from GPU
        if self._xp.__name__ == 'cupy':
            f = self._xp.asnumpy(f)
        #----------------------

        # convert complex128 arrays back to 2 float64 array
        f = real_view_of_complex_array(f)

        return f[..., 0]

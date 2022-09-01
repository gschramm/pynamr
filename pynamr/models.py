"""signal models for Na MR reconstruction"""
import typing
import abc
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


class DualTESodiumAcqModel(abc.ABC):
    """ abstract base class for decay models for dual TE Na MR data

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

    @abc.abstractmethod
    def forward(self):
        raise NotImplementedError

    @abc.abstractmethod
    def adjoint(self):
        raise NotImplementedError


class MonoExpDualTESodiumAcqModel(DualTESodiumAcqModel):

    def __init__(self, ds: int, sens: XpArray, dt: float,
                 readout_time: typing.Callable[[np.ndarray], np.ndarray],
                 kspace_part: RadialKSpacePartitioner) -> None:
        super().__init__(ds, sens, dt, readout_time, kspace_part)

    #------------------------------------------------------------------------------
    def forward(self, x: np.ndarray, gam: np.ndarray) -> np.ndarray:
        """ Calculate apodized FFT of an image f

            Parameters
            ----------

            x : a float64 numpy array of shape (self.image_shape,2)
              [...,0] is considered as the real part
              [...,1] is considered as the imag part

            gam : a float64 numpy/cupy array of shape (self.image_shape)

            Returns
            -------
            a float64 numpy array of shape (self.ncoils,self.image_shape,2)
        """

        # create a complex view of the input real input array with two channels
        x = complex_view_of_real_array(x)

        #----------------------
        # send f and gam to GPU
        if self._xp.__name__ == 'cupy':
            x = self._xp.asarray(x)
            gam = self._xp.asarray(gam)

        # downsample f and gamma
        x_ds = downsample(downsample(downsample(x, self._ds, axis=0),
                                     self._ds,
                                     axis=1),
                          self._ds,
                          axis=2)
        gam_ds = downsample(downsample(downsample(gam, self._ds, axis=0),
                                       self._ds,
                                       axis=1),
                            self._ds,
                            axis=2)

        y = self._xp.zeros((
            self._ncoils,
            2,
        ) + x_ds.shape,
                           dtype=self._xp.complex128)

        for i_sens in range(self._ncoils):
            for it in range(self.n_readout_bins):
                y[i_sens, 0, ...][self.readout_inds[it]] = self._xp.fft.fftn(
                    self.sens[i_sens, ...] *
                    gam_ds**(self._tr[it] / self._dt) * x_ds,
                    norm='ortho')[self.readout_inds[it]]
                y[i_sens, 1, ...][self.readout_inds[it]] = self._xp.fft.fftn(
                    self.sens[i_sens, ...] *
                    gam_ds**((self._tr[it] / self._dt) + 1) * x_ds,
                    norm='ortho')[self.readout_inds[it]]

        # get f, F, gam back from GPU
        if self._xp.__name__ == 'cupy':
            y = self._xp.asnumpy(y)
        #----------------------

        # convert complex128 arrays back to 2 float64 array
        y = real_view_of_complex_array(y)

        return y

    #------------------------------------------------------------------------------
    def adjoint(self, y: np.ndarray, gam: np.ndarray) -> np.ndarray:
        """ Calculate the adjoint of the apodized FFT of a k-space image F

            Parameters
            ----------

            y : a float64 numpy array of shape (self.ncoils,self.image_shape,2)
              [...,0] is considered as the real part
              [...,1] is considered as the imag part

            gam : a float64 numpy array of shape (self.image_shape)

            Returns
            -------
            a float64 numpy/cupy array of shape (self.image_shape)
        """
        # create a complex view of the input real input array with two channels
        y = complex_view_of_real_array(y)

        #----------------------
        # send F, gam to GPU
        if self._xp.__name__ == 'cupy':
            y = self._xp.asarray(y)
            gam = self._xp.asarray(gam)

        x_ds = self._xp.zeros(y.shape[2:], dtype=self._xp.complex128)

        gam_ds = downsample(downsample(downsample(gam, self._ds, axis=0),
                                       self._ds,
                                       axis=1),
                            self._ds,
                            axis=2)

        for i_sens in range(self._ncoils):
            for it in range(self.n_readout_bins):
                tmp0 = self._xp.zeros(y[i_sens, 0, ...].shape, dtype=y.dtype)
                tmp0[self.readout_inds[it]] = y[i_sens, 0,
                                                ...][self.readout_inds[it]]
                x_ds += (gam_ds**(self._tr[it] / self._dt)) * self._xp.conj(
                    self.sens[i_sens]) * self._xp.fft.ifftn(tmp0, norm='ortho')

                tmp1 = self._xp.zeros(y[i_sens, 1, ...].shape, dtype=y.dtype)
                tmp1[self.readout_inds[it]] = y[i_sens, 1,
                                                ...][self.readout_inds[it]]
                x_ds += (gam_ds**(
                    (self._tr[it] / self._dt) + 1)) * self._xp.conj(
                        self.sens[i_sens]) * self._xp.fft.ifftn(tmp1,
                                                                norm='ortho')

        # upsample f
        x = upsample(upsample(upsample(x_ds, self._ds, axis=0),
                              self._ds,
                              axis=1),
                     self._ds,
                     axis=2)

        # get f, F, gam back from GPU
        if self._xp.__name__ == 'cupy':
            x = self._xp.asnumpy(x)
        #----------------------

        # convert complex128 arrays back to 2 float64 array
        x = real_view_of_complex_array(x)

        return x

    #------------------------------------------------------------------------------

    def grad_gam(self, y: np.ndarray, gam: np.ndarray,
                 img: np.ndarray) -> np.ndarray:
        """ Calculate the the "inner" derivative with respect to gamma

            Parameters
            ----------

            y : a float64 numpy/cupy array of shape (self.ncoils,self.image_shape,2)
                containing the "outer derivative" of the cost function
              [...,0] is considered as the real part
              [...,1] is considered as the imag part

            gam : a float64 numpy/cupy array of shape (self.image_shape)

            img : a float64 numpy/cupy array of shape (self.image_shape,2)
                containing the current Na concentration image
              [...,0] is considered as the real part
              [...,1] is considered as the imag part

            Returns
            -------
            a float64 numpy/cupy array of shape (self.image_shape)
        """
        # create a complex view of the input real input array with two channels
        y = complex_view_of_real_array(y)
        img = complex_view_of_real_array(img)

        #----------------------
        # send y, gam to GPU
        if self._xp.__name__ == 'cupy':
            y = self._xp.asarray(y)
            gam = self._xp.asarray(gam)
            img = self._xp.asarray(img)

        x_ds = self._xp.zeros(y.shape[2:], dtype=self._xp.complex128)

        gam_ds = downsample(downsample(downsample(gam, self._ds, axis=0),
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

                tmp0 = self._xp.zeros(y[i_sens, 0, ...].shape, dtype=y.dtype)
                tmp0[self.readout_inds[it]] = y[i_sens, 0,
                                                ...][self.readout_inds[it]]
                x_ds += n * (gam_ds**(n - 1)) * self._xp.conj(
                    img_ds * self.sens[i_sens]) * self._xp.fft.ifftn(
                        tmp0, norm='ortho')

                tmp1 = self._xp.zeros(y[i_sens, 1, ...].shape, dtype=y.dtype)
                tmp1[self.readout_inds[it]] = y[i_sens, 1,
                                                ...][self.readout_inds[it]]
                x_ds += (n + 1) * (gam_ds**n) * self._xp.conj(
                    img_ds * self.sens[i_sens]) * self._xp.fft.ifftn(
                        tmp1, norm='ortho')

        # upsample f
        x = upsample(upsample(upsample(x_ds, self._ds, axis=0),
                              self._ds,
                              axis=1),
                     self._ds,
                     axis=2)

        # get f, F, gam back from GPU
        if self._xp.__name__ == 'cupy':
            x = self._xp.asnumpy(x)
        #----------------------

        # convert complex128 arrays back to 2 float64 array
        x = real_view_of_complex_array(x)

        return x[..., 0]

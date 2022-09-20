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
from .variables import Unknown, UnknownName

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------


class DualTESodiumAcqModel(abc.ABC):
    """ abstract base class for decay models for dual TE Na MR data """

    def __init__(self,
                 ds: int,
                 sens: XpArray,
                 dt: float,
                 readout_time: typing.Callable[[np.ndarray], np.ndarray],
                 kspace_part: RadialKSpacePartitioner) -> None:

        """ abstract base class for decay models for dual TE Na MR data

        Parameters
        ----------
        ds : int
            downsample factor (image shape / data shape)
        sens : XpArray
            complex array with coil sensitivities with shape (num_coils, data_shape)
        dt : float
            difference time between the two echos
        readout_time : typing.Callable[[np.ndarray], np.ndarray]
            Callable that maps 1d kspace array to readout time for used read out
        kspace_part : RadialKSpacePartitioner
            RadialKSpacePartioner that partitions cartesian k-space volume
            into radial shells of "same" readout time
        """

        # downsampling factor
        self._ds = ds
        # sensitivity "image" for each coil in downsampled data space
        self._sens = sens
        # number coils
        self._num_coils = self._sens.shape[0]

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

        # readout time of every shell
        self._tr = self._readout_time(self._kspace_part.k)


    @property
    def data_shape(self) -> tuple[int, int, int]:
        return self._data_shape

    @property
    def num_coils(self) -> int:
        return self._num_coils

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

    @property
    def y_shape_complex(self) -> tuple[int, int, int, int, int]:
        return (self.num_coils, ) + (2, ) + self.data_shape

    @property
    def y_shape_real(self) -> tuple[int, int, int, int, int, int]:
        return self.y_shape_complex + (2, )

    @abc.abstractmethod
    def forward(self, u: list[Unknown]) -> np.ndarray:
        """ forward model that maps from unknown images/parameters to data y

        Parameters
        ----------
        u : list[Unknown]
            the list of image space variables that represent known or unknown parameters of the forward model
            (e.g. images, image model parameters, T2* maps, Gamma)

        Returns
        -------
        np.ndarray
            the expected data with shape
            (num_coils, 2, data_shape, 2)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def adjoint(self, y:np.ndarray, u: list[Unknown]) -> np.ndarray:
        """ adjoint of forward model that maps from data y to image x

        Parameters
        ----------
        y : np.ndarray
            complex multicoil data array represented as real array with shape
            (num_coils, 2, data_shape, 2)

        u : list[Unknown]
            the list of image space variables that represent known or unknown parameters of the forward model
            (e.g. images, image model parameters, T2* maps, Gamma)

        Returns
        -------
        np.ndarray
            the (multichannel) complex image represented in a real array 
            with shape (num_compartments, image_shape, 2)
        """
        raise NotImplementedError


class TwoCompartmentBiExpDualTESodiumAcqModel(DualTESodiumAcqModel):
    """ two compartment dual echo sodium acquisition model with fixed T2star times """

    def __init__(
        self,
        ds: int,
        sens: XpArray,
        dt: float,
        readout_time: typing.Callable[[np.ndarray], np.ndarray],
        kspace_part: RadialKSpacePartitioner,
        T2star_free_short: float,
        T2star_free_long: float,
        T2star_bound_short: float,
        T2star_bound_long: float,
        free_long_frac: float,
        bound_long_frac: float,
    ) -> None:
        """ two compartment dual echo sodium acquisition model with fixed T2star times

        Parameters
        ----------
        ds : int
            downsample factor (image shape / data shape)
        sens : XpArray
            complex array with coil sensitivities with shape (num_coils, data_shape)
        dt : float
            difference time between the two echos
        readout_time : typing.Callable[[np.ndarray], np.ndarray]
            Callable that maps 1d kspace array to readout time for used read out
        kspace_part : RadialKSpacePartitioner
            RadialKSpacePartioner that partitions cartesian k-space volume
            into radial shells of "same" readout time
        T2star_free_short : float
            short T2start time of free compartment
        T2star_free_long : float
            long T2start time of free compartment
        T2star_bound_short : float
            short T2start time of bound compartment
        T2star_bound_long : float
            long T2start time of bound compartment
        free_long_frac : float
            fraction of free compartment undergoing long/slow decay
        bound_long_frac : float
            fraction of bound compartment undergoing long/slow decay
        """

        super().__init__(ds,
                         sens,
                         dt,
                         readout_time,
                         kspace_part)

        self._T2star_free_short = T2star_free_short
        self._T2star_free_long = T2star_free_long
        self._T2star_bound_short = T2star_bound_short
        self._T2star_bound_long = T2star_bound_long

        self._free_long_frac = free_long_frac
        self._bound_long_frac = bound_long_frac

        self._free_short_frac = 1 - self._free_long_frac
        self._bound_short_frac = 1 - self._bound_long_frac
        

    @property
    def T2star_free_short(self) -> float:
        return self._T2star_free_short

    @property
    def T2star_free_long(self) -> float:
        return self._T2star_free_long

    @property
    def T2star_bound_short(self) -> float:
        return self._T2star_bound_short

    @property
    def T2star_bound_long(self) -> float:
        return self._T2star_bound_long

    @property
    def free_long_frac(self) -> float:
        return self._free_long_frac

    @property
    def bound_long_frac(self) -> float:
        return self._bound_long_frac

    @property
    def free_short_frac(self) -> float:
        return self._free_short_frac

    @property
    def bound_short_frac(self) -> float:
        return self._bound_short_frac

    def forward(self, u: list[Unknown]) -> np.ndarray:
        """ forward step that calculates expected signal

        Parameters
        ----------
        u : list[Unknown]
            the list of image space variables that represent known or unknown parameters of the forward model,
            here single image variable with 2 compartments (bound and free pool)

        Returns
        -------
        np.ndarray (real)
            the expected signal in all channels
            shape (num_coils, data_shape, 2)
        """ """"""

        # create a complex view of the input real input array with two channels
        x = complex_view_of_real_array(u[0]._value)

        #----------------------
        # send f and gam to GPU
        if self._xp.__name__ == 'cupy':
            x = self._xp.asarray(x)

        # downsample the image along the spatial dimensions
        x_ds = downsample(downsample(downsample(x, self._ds, axis=1),
                                     self._ds,
                                     axis=2),
                          self._ds,
                          axis=3)

        y = self._xp.zeros(self.y_shape_complex, dtype=self._xp.complex128)

        for i_sens in range(self._num_coils):
            for it in range(self.n_readout_bins):
                if self._bound_long_frac > 0:
                    # first echo - bound compartment - long
                    y[i_sens, 0, ...][self.readout_inds[
                        it]] += self._bound_long_frac * self._xp.fft.fftn(
                            self.sens[i_sens, ...] * self._xp.exp(
                                -self._tr[it] / self._T2star_bound_long) *
                            x_ds[0, ...],
                            norm='ortho')[self.readout_inds[it]]
                    # second echo - bound compartment - long
                    y[i_sens, 1, ...][self.readout_inds[
                        it]] += self._bound_long_frac * self._xp.fft.fftn(
                            self.sens[i_sens, ...] * self._xp.exp(
                                -(self._tr[it] + self._dt) /
                                self._T2star_bound_long) * x_ds[0, ...],
                            norm='ortho')[self.readout_inds[it]]

                if self._bound_short_frac > 0:
                    # first echo - bound compartment - short
                    y[i_sens, 0, ...][self.readout_inds[
                        it]] += self._bound_short_frac * self._xp.fft.fftn(
                            self.sens[i_sens, ...] * self._xp.exp(
                                -self._tr[it] / self._T2star_bound_short) *
                            x_ds[0, ...],
                            norm='ortho')[self.readout_inds[it]]
                    # second echo - bound compartment - short
                    y[i_sens, 1, ...][self.readout_inds[
                        it]] += self._bound_short_frac * self._xp.fft.fftn(
                            self.sens[i_sens, ...] * self._xp.exp(
                                -(self._tr[it] + self._dt) /
                                self._T2star_bound_short) * x_ds[0, ...],
                            norm='ortho')[self.readout_inds[it]]

                if self._free_long_frac > 0:
                    # first echo - free compartment - long
                    y[i_sens, 0, ...][self.readout_inds[
                        it]] += self._free_long_frac * self._xp.fft.fftn(
                            self.sens[i_sens, ...] * self._xp.exp(
                                -self._tr[it] / self._T2star_free_long) *
                            x_ds[1, ...],
                            norm='ortho')[self.readout_inds[it]]
                    # second echo - free compartment - long
                    y[i_sens, 1, ...][self.readout_inds[
                        it]] += self._free_long_frac * self._xp.fft.fftn(
                            self.sens[i_sens, ...] * self._xp.exp(
                                -(self._tr[it] + self._dt) /
                                self._T2star_free_long) * x_ds[1, ...],
                            norm='ortho')[self.readout_inds[it]]

                if self._free_short_frac > 0:
                    # first echo - free compartment - short
                    y[i_sens, 0, ...][self.readout_inds[
                        it]] += self._free_short_frac * self._xp.fft.fftn(
                            self.sens[i_sens, ...] * self._xp.exp(
                                -self._tr[it] / self._T2star_free_short) *
                            x_ds[1, ...],
                            norm='ortho')[self.readout_inds[it]]
                    # second echo - free compartment - short
                    y[i_sens, 1, ...][self.readout_inds[
                        it]] += self._free_short_frac * self._xp.fft.fftn(
                            self.sens[i_sens, ...] * self._xp.exp(
                                -(self._tr[it] + self._dt) /
                                self._T2star_free_short) * x_ds[1, ...],
                            norm='ortho')[self.readout_inds[it]]

        # get y back from GPU
        if self._xp.__name__ == 'cupy':
            y = self._xp.asnumpy(y)
        #----------------------

        # convert complex128 arrays back to 2 float64 array
        y = real_view_of_complex_array(y)

        return y

    def adjoint(self, y: np.ndarray, u:list[Unknown]) -> np.ndarray:
        """ adjoint of forward step 

        Parameters
        ----------
        y : np.ndarray (real)
            a "signal" in data space of shape
            (num_coils, data_shape, 2)

        Returns
        -------
        np.ndarray (real)
            two compartment image of shape (2,n0,n1,n2,2)
            first (left most) dimension are the "channels"
            0 ... bound 
            1 ... free
        """

        # create a complex view of the input real input array with two channels
        y = complex_view_of_real_array(y)

        #----------------------
        # send y, gam to GPU
        if self._xp.__name__ == 'cupy':
            y = self._xp.asarray(y)

        x_ds = self._xp.zeros((2,)+self._data_shape, dtype=y.dtype)

        for i_sens in range(self._num_coils):
            for it in range(self.n_readout_bins):
                if self._bound_long_frac > 0:
                    # first echo - bound compartment - long
                    tmp = self._xp.zeros(y[i_sens, 0, ...].shape,
                                         dtype=y.dtype)
                    tmp[self.readout_inds[it]] = y[i_sens, 0,
                                                   ...][self.readout_inds[it]]
                    x_ds[0, ...] += self._bound_long_frac * (self._xp.exp(
                        -self._tr[it] /
                        self._T2star_bound_long)) * self._xp.conj(
                            self.sens[i_sens]) * self._xp.fft.ifftn(
                                tmp, norm='ortho')
                    # second echo - bound compartment - long
                    tmp = self._xp.zeros(y[i_sens, 1, ...].shape,
                                         dtype=y.dtype)
                    tmp[self.readout_inds[it]] = y[i_sens, 1,
                                                   ...][self.readout_inds[it]]
                    x_ds[0, ...] += self._bound_long_frac * (self._xp.exp(
                        -(self._tr[it] + self._dt) /
                        self._T2star_bound_long)) * self._xp.conj(
                            self.sens[i_sens]) * self._xp.fft.ifftn(
                                tmp, norm='ortho')

                if self._bound_short_frac > 0:
                    # first echo - bound compartment - long
                    tmp = self._xp.zeros(y[i_sens, 0, ...].shape,
                                         dtype=y.dtype)
                    tmp[self.readout_inds[it]] = y[i_sens, 0,
                                                   ...][self.readout_inds[it]]
                    x_ds[0, ...] += self._bound_short_frac * (self._xp.exp(
                        -self._tr[it] /
                        self._T2star_bound_short)) * self._xp.conj(
                            self.sens[i_sens]) * self._xp.fft.ifftn(
                                tmp, norm='ortho')
                    # second echo - bound compartment - long
                    tmp = self._xp.zeros(y[i_sens, 1, ...].shape,
                                         dtype=y.dtype)
                    tmp[self.readout_inds[it]] = y[i_sens, 1,
                                                   ...][self.readout_inds[it]]
                    x_ds[0, ...] += self._bound_short_frac * (self._xp.exp(
                        -(self._tr[it] + self._dt) /
                        self._T2star_bound_short)) * self._xp.conj(
                            self.sens[i_sens]) * self._xp.fft.ifftn(
                                tmp, norm='ortho')

                if self._free_long_frac > 0:
                    # first echo - bound compartment - long
                    tmp = self._xp.zeros(y[i_sens, 0, ...].shape,
                                         dtype=y.dtype)
                    tmp[self.readout_inds[it]] = y[i_sens, 0,
                                                   ...][self.readout_inds[it]]
                    x_ds[1, ...] += self._free_long_frac * (self._xp.exp(
                        -self._tr[it] /
                        self._T2star_free_long)) * self._xp.conj(
                            self.sens[i_sens]) * self._xp.fft.ifftn(
                                tmp, norm='ortho')
                    # second echo - bound compartment - long
                    tmp = self._xp.zeros(y[i_sens, 1, ...].shape,
                                         dtype=y.dtype)
                    tmp[self.readout_inds[it]] = y[i_sens, 1,
                                                   ...][self.readout_inds[it]]
                    x_ds[1, ...] += self._free_long_frac * (self._xp.exp(
                        -(self._tr[it] + self._dt) /
                        self._T2star_free_long)) * self._xp.conj(
                            self.sens[i_sens]) * self._xp.fft.ifftn(
                                tmp, norm='ortho')

                if self._free_short_frac > 0:
                    # first echo - bound compartment - long
                    tmp = self._xp.zeros(y[i_sens, 0, ...].shape,
                                         dtype=y.dtype)
                    tmp[self.readout_inds[it]] = y[i_sens, 0,
                                                   ...][self.readout_inds[it]]
                    x_ds[1, ...] += self._free_short_frac * (self._xp.exp(
                        -self._tr[it] /
                        self._T2star_free_short)) * self._xp.conj(
                            self.sens[i_sens]) * self._xp.fft.ifftn(
                                tmp, norm='ortho')
                    # second echo - bound compartment - long
                    tmp = self._xp.zeros(y[i_sens, 1, ...].shape,
                                         dtype=y.dtype)
                    tmp[self.readout_inds[it]] = y[i_sens, 1,
                                                   ...][self.readout_inds[it]]
                    x_ds[1, ...] += self._free_short_frac * (self._xp.exp(
                        -(self._tr[it] + self._dt) /
                        self._T2star_free_short)) * self._xp.conj(
                            self.sens[i_sens]) * self._xp.fft.ifftn(
                                tmp, norm='ortho')

        # upsample x
        x = upsample(upsample(upsample(x_ds, self._ds, axis=1),
                              self._ds,
                              axis=2),
                     self._ds,
                     axis=3)

        # get x gam back from GPU
        if self._xp.__name__ == 'cupy':
            x = self._xp.asnumpy(x)
        #----------------------

        # convert complex128 arrays back to 2 float64 array
        x = real_view_of_complex_array(x)

        return x


class MonoExpDualTESodiumAcqModel(DualTESodiumAcqModel):
    """ mono exponential dual TE sodium acquisition model assuming one compartment """

    def __init__(self,
                 ds: int,
                 sens: XpArray,
                 dt: float,
                 readout_time: typing.Callable[[np.ndarray], np.ndarray],
                 kspace_part: RadialKSpacePartitioner ) -> None:
        """ mono exponential dual TE sodium acquisition model assuming one compartment

        Parameters
        ----------
        ds : int
            downsample factor (image shape / data shape)
        sens : XpArray
            complex array with coil sensitivities with shape (num_coils, data_shape)
        dt : float
            difference time between the two echos
        readout_time : typing.Callable[[np.ndarray], np.ndarray]
            Callable that maps 1d kspace array to readout time for used read out
        kspace_part : RadialKSpacePartitioner
            RadialKSpacePartioner that partitions cartesian k-space volume
            into radial shells of "same" readout time
        """
        super().__init__(ds,
                         sens,
                         dt,
                         readout_time,
                         kspace_part)

    #------------------------------------------------------------------------------
    def forward(self, u: list[Unknown]) -> np.ndarray:
        """ Calculate apodized FFT of an image f

            Parameters
            ----------

            u : list[Unknown]
                the list of image space variables that represent known or unknown parameters of the forward model,
                here two variables
                1) the image proportional to Na concentration, with real and imaginary parts (last dimension)
                2) the gamma, monoexponential T2* decay from TE1 to TE2, real

            Returns
            -------
            a float64 numpy array of shape (self.num_coils,image_shape,2)
        """
        # read the input variables and create a complex view if required
        for el in u:
            if el._name==UnknownName.IMAGE:
                x = complex_view_of_real_array(el._value)
            if el._name==UnknownName.GAMMA:
                gam = el._value

        #----------------------
        # send x and gam to GPU
        if self._xp.__name__ == 'cupy':
            x = self._xp.asarray(x)
            gam = self._xp.asarray(gam)

        # downsample x and gamma
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

        y = self._xp.zeros(self.y_shape_complex, dtype=self._xp.complex128)


        for i_sens in range(self._num_coils):
            for it in range(self.n_readout_bins):
                y[i_sens, 0, ...][self.readout_inds[it]] = self._xp.fft.fftn(
                    self.sens[i_sens, ...] *
                    gam_ds**(self._tr[it] / self._dt) * x_ds,
                    norm='ortho')[self.readout_inds[it]]
                y[i_sens, 1, ...][self.readout_inds[it]] = self._xp.fft.fftn(
                    self.sens[i_sens, ...] *
                    gam_ds**((self._tr[it] / self._dt) + 1) * x_ds,
                    norm='ortho')[self.readout_inds[it]]

        # get x from GPU
        if self._xp.__name__ == 'cupy':
            y = self._xp.asnumpy(y)
        #----------------------

        # convert complex128 arrays back to 2 float64 array
        y = real_view_of_complex_array(y)

        return y

    #------------------------------------------------------------------------------
    def adjoint(self, y: np.ndarray, u: list[Unknown]) -> np.ndarray:
        """ Calculate the adjoint of the apodized FFT of a k-space image F

            Parameters
            ----------

            y : a float64 numpy array of shape (self.num_coils,2,self.image_shape,2)
              [...,0] is considered as the real part
              [...,1] is considered as the imag part

            u : list[Unknown]
                the list of image space variables that represent known or unknown parameters of the forward model,
                here two variables
                1) the image proportional to Na concentration, with real and imaginary parts (last dimension)
                2) the gamma, monoexponential T2* decay from TE1 to TE2, real

            Returns
            -------
            a float64 numpy array of shape (1,self.image_shape,2)
        """
        # create a complex view of the input real input array with two channels
        y = complex_view_of_real_array(y)

        for el in u:
            if el._name==UnknownName.GAMMA:
                gam = el._value

        #----------------------
        # send y, gam to GPU
        if self._xp.__name__ == 'cupy':
            y = self._xp.asarray(y)
            gam = self._xp.asarray(gam)

        x_ds = self._xp.zeros(self.data_shape,
                              dtype=self._xp.complex128)

        gam_ds = downsample(downsample(downsample(gam, self._ds, axis=0),
                                       self._ds,
                                       axis=1),
                            self._ds,
                            axis=2)

        for i_sens in range(self._num_coils):
            for it in range(self.n_readout_bins):
                # 1st echo
                tmp0 = self._xp.zeros(y[i_sens, 0, ...].shape, dtype=y.dtype)
                tmp0[self.readout_inds[it]] = y[i_sens, 0,
                                                ...][self.readout_inds[it]]
                x_ds += (gam_ds
                                 **(self._tr[it] / self._dt)) * self._xp.conj(
                                     self.sens[i_sens]) * self._xp.fft.ifftn(
                                         tmp0, norm='ortho')

                # 2nd echo
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

        # get x gam back from GPU
        if self._xp.__name__ == 'cupy':
            x = self._xp.asnumpy(x)
        #----------------------

        # convert complex128 arrays back to 2 float64 array
        x = real_view_of_complex_array(x)

        return x

    #------------------------------------------------------------------------------

    def grad(self, y: np.ndarray, u: list[Unknown]) -> np.ndarray:
        """ Calculate the the "inner" derivative with respect to gamma

            Parameters
            ----------

            y : a float64 numpy array of shape (self.num_coils,2,self.image_shape,2)
                containing the "outer derivative" of the cost function
              [...,0] is considered as the real part
              [...,1] is considered as the imag part

            u : list[Unknown]
                the list of image space variables that represent known or unknown parameters of the forward model,
                here two variables
                1) the image proportional to Na concentration, with real and imaginary parts (last dimension)
                2) the gamma, monoexponential T2* decay from TE1 to TE2, real

            Returns
            -------
            a float64 numpy array of shape (1,image_shape)
        """

        # currently only for Gamma
        if u[0]._name != UnknownName.GAMMA:
            raise NotImplementedError('Currently only for nonlinear variables (Gamma)')

        # create a complex view of the input real input array with two channels
        y = complex_view_of_real_array(y)

        for el in u:
            if el._name==UnknownName.IMAGE:
                img = complex_view_of_real_array(el._value)
            elif el._name==UnknownName.GAMMA:
                gam = el._value

        #----------------------
        # send y, gam to GPU
        if self._xp.__name__ == 'cupy':
            y = self._xp.asarray(y)
            gam = self._xp.asarray(gam)
            img = self._xp.asarray(img)

        x_ds = self._xp.zeros(self.data_shape,
                              dtype=self._xp.complex128)

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

        for i_sens in range(self._num_coils):
            for it in range(self.n_readout_bins):
                n = self._tr[it] / self._dt

                # 1st echo
                tmp0 = self._xp.zeros(y[i_sens, 0, ...].shape, dtype=y.dtype)
                tmp0[self.readout_inds[it]] = y[i_sens, 0,
                                                ...][self.readout_inds[it]]
                x_ds += n * (gam_ds**(n - 1)) * self._xp.conj(
                    img_ds.squeeze() * self.sens[i_sens]) * self._xp.fft.ifftn(
                        tmp0, norm='ortho')

                # 2nd echo
                tmp1 = self._xp.zeros(y[i_sens, 1, ...].shape, dtype=y.dtype)
                tmp1[self.readout_inds[it]] = y[i_sens, 1,
                                                ...][self.readout_inds[it]]
                x_ds += (n + 1) * (gam_ds**n) * self._xp.conj(
                    img_ds.squeeze() * self.sens[i_sens]) * self._xp.fft.ifftn(
                        tmp1, norm='ortho')

        # upsample x
        x = upsample(upsample(upsample(x_ds, self._ds, axis=0),
                              self._ds,
                              axis=1),
                     self._ds,
                     axis=2)

        # get x gam back from GPU
        if self._xp.__name__ == 'cupy':
            x = self._xp.asnumpy(x)
        #----------------------

        # convert complex128 arrays back to 2 float64 array
        x = real_view_of_complex_array(x)

        return x[..., 0]

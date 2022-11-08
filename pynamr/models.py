""" Forward models for Na MRI reconstruction

    Implementatation details: not everything is done with complex arrays and cupy because of scipy.optimize.minimize 

"""
import typing
import abc
import numpy as np
import sys

# check whether cupy is available
try:
    import cupy as cp
except ModuleNotFoundError:
    import numpy as cp

from .utils import RadialKSpacePartitioner, XpArray
from .utils import complex_view_of_real_array, real_view_of_complex_array
from .utils import downsample, downsample_transpose
from .variables import Var, VarName

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------


class DualTESodiumAcqModel(abc.ABC):
    """ abstract base class for decay models for dual TE Na MR data """

    def __init__(self,
                 ds: int,
                 sens: np.ndarray,
                 dt: float,
                 te1: float,
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
        te1 : float
            TE1, start time for the first acquisition
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

        # TE1
        self._te1 = te1

        # numpy / cupy module to use for ffts
        if 'cupy' in sys.modules:
            self._xp = cp
            self._sens = self._xp.asarray(self._sens)
        else:
            self._xp = np

        # callable that calculates readout time from k array
        self._readout_time = readout_time

        # object that partions cartesian k-space into radial shells
        self._kspace_part = kspace_part
        # shape of the data
        self._data_shape = self._kspace_part._data_shape

        # readout time of every shell
        self._tr = self._readout_time(self._kspace_part.k)

        # T2* value below which we can assume that the signal is completely lost
        # for this pulse sequence
        self._t2_zero = self._te1 * 0.1


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
    def te1(self) -> float:
        return self._te1

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

    @property
    def t2_zero(self) -> float:
        return self._t2_zero

    def safe_decay(self, time: float, t2: float | XpArray) -> float | XpArray:
        """ utility function for computing the T2* decay, with safety net for very low T2* values

        Parameters
        ----------
        time : decay time
        t2 : T2* relaxation time, either scalar or spatial map

        Returns
        -------
        float or XpArray
            the multiplicative factor that represents the exponential decay
        """
        if self._xp.isscalar(t2):
            temp = self._xp.exp( -time / t2) if t2 > self._t2_zero else 0.
        else:
            temp = self._xp.zeros(t2.shape, self._xp.float64)
            temp[t2 > self._t2_zero] = self._xp.exp( -time / t2[t2 > self._t2_zero])
        return temp


    @abc.abstractmethod
    def forward(self, var_dict: dict[VarName,Var]) -> np.ndarray:
        """ forward model that maps from images/parameters to data y

        Parameters
        ----------
        u : dict[VarName,Var]
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
    def adjoint(self, y:np.ndarray, var_dict: dict[VarName,Var]) -> np.ndarray:
        """ adjoint of forward model that maps from data y to image x

        Parameters
        ----------
        y : np.ndarray
            complex multicoil data array represented as real array with shape
            (num_coils, 2, data_shape, 2)

        u : dict[VarName,Var]
            the list of image space variables that represent known or unknown parameters of the forward model
            (e.g. images, image model parameters, T2* maps, Gamma)

        Returns
        -------
        np.ndarray of image shape
        """
        raise NotImplementedError


class TwoCompartmentBiExpDualTESodiumAcqModel(DualTESodiumAcqModel):
    """ two compartment dual echo sodium acquisition model with fixed T2star times """

    def __init__(
        self,
        ds: int,
        sens: XpArray,
        dt: float,
        te1: float,
        readout_time: typing.Callable[[np.ndarray], np.ndarray],
        kspace_part: RadialKSpacePartitioner,
        T2star_free_short: float | np.ndarray,
        T2star_free_long: float | np.ndarray,
        T2star_bound_short: float | np.ndarray,
        T2star_bound_long: float | np.ndarray,
        free_long_frac: float | np.ndarray,
        bound_long_frac: float | np.ndarray,
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
        te1 : float
            TE1, start time for the first acquisition
        readout_time : typing.Callable[[np.ndarray], np.ndarray]
            Callable that maps 1d kspace array to readout time for used read out
        kspace_part : RadialKSpacePartitioner
            RadialKSpacePartioner that partitions cartesian k-space volume
            into radial shells of "same" readout time
        T2star_free_short : float | np.ndarray
            short T2start time of free compartment, either scalar or spatial map
        T2star_free_long : float | np.ndarray
            long T2start time of free compartment, either scalar or spatial map
        T2star_bound_short : float | np.ndarray
            short T2start time of bound compartment, either scalar or spatial map
        T2star_bound_long : float | np.ndarray
            long T2start time of bound compartment, either scalar or spatial map
        free_long_frac : float | np.ndarray
            fraction of free compartment undergoing long/slow decay, either scalar or spatial map
        bound_long_frac : float | np.ndarray
            fraction of bound compartment undergoing long/slow decay, either scalar or spatial map
        """

        super().__init__(ds,
                         sens,
                         dt,
                         te1,
                         readout_time,
                         kspace_part)

        # if T2* spatial maps and using cupy, convert immediately for convenience to cupy arrays
        if self._xp.__name__ == 'cupy' and not self._xp.isscalar(T2star_bound_short):
            self._T2star_free_short = self._xp.asarray(T2star_free_short)
            self._T2star_free_long = self._xp.asarray(T2star_free_long)
            self._T2star_bound_short = self._xp.asarray(T2star_bound_short)
            self._T2star_bound_long = self._xp.asarray(T2star_bound_long)

            self._free_long_frac = self._xp.asarray(free_long_frac)
            self._bound_long_frac = self._xp.asarray(bound_long_frac)
        else:
            self._T2star_free_short = T2star_free_short
            self._T2star_free_long = T2star_free_long
            self._T2star_bound_short = T2star_bound_short
            self._T2star_bound_long = T2star_bound_long

            self._free_long_frac = free_long_frac
            self._bound_long_frac = bound_long_frac

        self._free_short_frac = 1 - self._free_long_frac
        self._bound_short_frac = 1 - self._bound_long_frac 


    def forward(self, var_dict: dict[VarName,Var]) -> np.ndarray:
        """ forward step that calculates expected signal

        Parameters
        ----------
        u : dict[VarName,Var]
            the list of image space variables that represent known or unknown parameters of the forward model,
            -> here single complex image variable with shape (2 - compartments, 3D spatial dimensions, 2 - real and imaginary dimensions)

        Returns
        -------
        np.ndarray (real)
            the expected signal in all channels
            shape (num_coils, data_shape, 2)
        """ """"""

        # create a complex view of the input real input array with two channels
        x = complex_view_of_real_array(var_dict[VarName.PARAM].value)

        #----------------------
        # send f and gam to GPU
        if self._xp.__name__ == 'cupy':
            x = self._xp.asarray(x) 

        y = self._xp.zeros(self.y_shape_complex, dtype=self._xp.complex128)

        for i_sens in range(self._num_coils):
            for it in range(self.n_readout_bins):
                    # first echo - bound compartment - long
                    temp = self._bound_long_frac * self.sens[i_sens, ...] \
                            * self.safe_decay( self._tr[it] + self._te1, self._T2star_bound_long) \
                            * x[0, ...]
                    temp = downsample(downsample(downsample(temp, self._ds, axis=0),
                                                            self._ds,
                                                            axis=1),
                                                 self._ds,
                                                 axis=2)
                    y[i_sens, 0, ...][self.readout_inds[
                        it]] += self._xp.fft.fftn(temp,
                            norm='ortho')[self.readout_inds[it]]
                    # second echo - bound compartment - long
                    temp = self._bound_long_frac * self.sens[i_sens, ...] \
                    * self.safe_decay( self._tr[it] + self._te1 + self._dt, self._T2star_bound_long) \
                    * x[0, ...]
                    temp = downsample(downsample(downsample(temp, self._ds, axis=0),
                                                            self._ds,
                                                            axis=1),
                                                 self._ds,
                                                 axis=2)
                    y[i_sens, 1, ...][self.readout_inds[
                        it]] += self._xp.fft.fftn(temp,
                            norm='ortho')[self.readout_inds[it]]

                    # first echo - bound compartment - short
                    temp = self._bound_short_frac * self.sens[i_sens, ...] \
                    * self.safe_decay( self._tr[it] + self._te1, self._T2star_bound_short) \
                    * x[0, ...]
                    temp = downsample(downsample(downsample(temp, self._ds, axis=0),
                                                            self._ds,
                                                            axis=1),
                                                 self._ds,
                                                 axis=2)
                    y[i_sens, 0, ...][self.readout_inds[
                        it]] += self._xp.fft.fftn(temp,
                                norm='ortho')[self.readout_inds[it]]
                    # second echo - bound compartment - short
                    temp = self._bound_short_frac * self.sens[i_sens, ...] \
                    * self.safe_decay( self._tr[it] + self._te1 + self._dt, self._T2star_bound_short) \
                    * x[0, ...]
                    temp = downsample(downsample(downsample(temp, self._ds, axis=0),
                                                            self._ds,
                                                            axis=1),
                                                 self._ds,
                                                 axis=2)
                    y[i_sens, 1, ...][self.readout_inds[
                        it]] += self._xp.fft.fftn(temp,
                            norm='ortho')[self.readout_inds[it]]

                    # first echo - free compartment - long
                    temp = self._free_long_frac * self.sens[i_sens, ...] \
                    * self.safe_decay( self._tr[it] + self._te1, self._T2star_free_long) \
                    * x[1, ...]
                    temp = downsample(downsample(downsample(temp, self._ds, axis=0),
                                                            self._ds,
                                                            axis=1),
                                                 self._ds,
                                                 axis=2)
                    y[i_sens, 0, ...][self.readout_inds[
                        it]] += self._xp.fft.fftn(temp,
                            norm='ortho')[self.readout_inds[it]]
                    # second echo - free compartment - long
                    temp = self._free_long_frac * self.sens[i_sens, ...] \
                    * self.safe_decay( self._tr[it] + self._te1 + self._dt, self._T2star_free_long) \
                    * x[1, ...]
                    temp = downsample(downsample(downsample(temp, self._ds, axis=0),
                                                            self._ds,
                                                            axis=1),
                                                 self._ds,
                                                 axis=2)
                    y[i_sens, 1, ...][self.readout_inds[
                        it]] += self._xp.fft.fftn(temp,
                            norm='ortho')[self.readout_inds[it]]

                    # first echo - free compartment - short
                    temp = self._free_short_frac * self.sens[i_sens, ...] \
                    * self.safe_decay( self._tr[it] + self._te1, self._T2star_free_short) \
                    * x[1, ...]
                    temp = downsample(downsample(downsample(temp, self._ds, axis=0),
                                                            self._ds,
                                                            axis=1),
                                                 self._ds,
                                                 axis=2)
                    y[i_sens, 0, ...][self.readout_inds[
                        it]] += self._xp.fft.fftn(temp,
                            norm='ortho')[self.readout_inds[it]]
                    # second echo - free compartment - short
                    temp = self._free_short_frac * self.sens[i_sens, ...] \
                    * self.safe_decay( self._tr[it] + self._te1 + self._dt, self._T2star_free_short) \
                    * x[1, ...]
                    temp = downsample(downsample(downsample(temp, self._ds, axis=0),
                                                            self._ds,
                                                            axis=1),
                                                 self._ds,
                                                 axis=2)
                    y[i_sens, 1, ...][self.readout_inds[
                        it]] += self._xp.fft.fftn(temp,
                            norm='ortho')[self.readout_inds[it]]

        # get y back from GPU
        if self._xp.__name__ == 'cupy':
            y = self._xp.asnumpy(y)
        #----------------------

        # convert complex128 arrays back to 2 float64 array
        y = real_view_of_complex_array(y)

        return y

    def adjoint(self, y: np.ndarray, var_dict: dict[VarName,Var]) -> np.ndarray:
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

        x = self._xp.zeros(var_dict[VarName.PARAM].complex_shape, dtype=y.dtype)

        for i_sens in range(self._num_coils):
            for it in range(self.n_readout_bins):
                    # first echo - bound compartment - long
                    tmp = self._xp.zeros(y[i_sens, 0, ...].shape,
                                         dtype=y.dtype)
                    tmp[self.readout_inds[it]] = y[i_sens, 0,
                                                   ...][self.readout_inds[it]]

                    tmp = self._xp.fft.ifftn(tmp, norm='ortho')
                    tmp = downsample_transpose(downsample_transpose(downsample_transpose(tmp, self._ds, axis=0),
                                                                    self._ds,
                                                                    axis=1),
                                               self._ds,
                                               axis=2)

                    x[0, ...] += self._bound_long_frac * self.safe_decay( self._tr[it] + self._te1, self._T2star_bound_long) \
                                * self._xp.conj(self.sens[i_sens]) * tmp
                    # second echo - bound compartment - long
                    tmp = self._xp.zeros(y[i_sens, 1, ...].shape,
                                         dtype=y.dtype)
                    tmp[self.readout_inds[it]] = y[i_sens, 1,
                                                   ...][self.readout_inds[it]]
                    tmp = self._xp.fft.ifftn(tmp, norm='ortho')
                    tmp = downsample_transpose(downsample_transpose(downsample_transpose(tmp, self._ds, axis=0),
                                                                    self._ds,
                                                                    axis=1),
                                               self._ds,
                                               axis=2)
                    x[0, ...] += self._bound_long_frac * self.safe_decay( self._tr[it] + self._te1 + self._dt, self._T2star_bound_long) \
                                * self._xp.conj(self.sens[i_sens]) * tmp

                    # first echo - bound compartment - long
                    tmp = self._xp.zeros(y[i_sens, 0, ...].shape,
                                         dtype=y.dtype)
                    tmp[self.readout_inds[it]] = y[i_sens, 0,
                                                   ...][self.readout_inds[it]]
                    tmp = self._xp.fft.ifftn(tmp, norm='ortho')
                    tmp = downsample_transpose(downsample_transpose(downsample_transpose(tmp, self._ds, axis=0),
                                                                    self._ds,
                                                                    axis=1),
                                               self._ds,
                                               axis=2)
                    x[0, ...] += self._bound_short_frac * self.safe_decay( self._tr[it] + self._te1, self._T2star_bound_short) \
                                * self._xp.conj(self.sens[i_sens]) * tmp
                    # second echo - bound compartment - long
                    tmp = self._xp.zeros(y[i_sens, 1, ...].shape,
                                         dtype=y.dtype)
                    tmp[self.readout_inds[it]] = y[i_sens, 1,
                                                   ...][self.readout_inds[it]]
                    tmp = self._xp.fft.ifftn(tmp, norm='ortho')
                    tmp = downsample_transpose(downsample_transpose(downsample_transpose(tmp, self._ds, axis=0),
                                                                    self._ds,
                                                                    axis=1),
                                               self._ds,
                                               axis=2)
                    x[0, ...] += self._bound_short_frac * self.safe_decay( self._tr[it] + self._te1 + self._dt, self._T2star_bound_short) \
                                    * self._xp.conj(self.sens[i_sens]) * tmp

                    # first echo - bound compartment - long
                    tmp = self._xp.zeros(y[i_sens, 0, ...].shape,
                                         dtype=y.dtype)
                    tmp[self.readout_inds[it]] = y[i_sens, 0,
                                                   ...][self.readout_inds[it]]
                    tmp = self._xp.fft.ifftn(tmp, norm='ortho')
                    tmp = downsample_transpose(downsample_transpose(downsample_transpose(tmp, self._ds, axis=0),
                                                                    self._ds,
                                                                    axis=1),
                                               self._ds,
                                               axis=2)
                    x[1, ...] += self._free_long_frac * self.safe_decay( self._tr[it] + self._te1, self._T2star_free_long) \
                                    * self._xp.conj(self.sens[i_sens]) * tmp
                    # second echo - bound compartment - long
                    tmp = self._xp.zeros(y[i_sens, 1, ...].shape,
                                         dtype=y.dtype)
                    tmp[self.readout_inds[it]] = y[i_sens, 1,
                                                   ...][self.readout_inds[it]]
                    tmp = self._xp.fft.ifftn(tmp, norm='ortho')
                    tmp = downsample_transpose(downsample_transpose(downsample_transpose(tmp, self._ds, axis=0),
                                                                    self._ds,
                                                                    axis=1),
                                               self._ds,
                                               axis=2)
                    x[1, ...] += self._free_long_frac * self.safe_decay( self._tr[it] + self._te1 + self._dt, self._T2star_free_long) \
                                    * self._xp.conj(self.sens[i_sens]) * tmp

                    # first echo - bound compartment - long
                    tmp = self._xp.zeros(y[i_sens, 0, ...].shape,
                                         dtype=y.dtype)
                    tmp[self.readout_inds[it]] = y[i_sens, 0,
                                                   ...][self.readout_inds[it]]
                    tmp = self._xp.fft.ifftn(tmp, norm='ortho')
                    tmp = downsample_transpose(downsample_transpose(downsample_transpose(tmp, self._ds, axis=0),
                                                                    self._ds,
                                                                    axis=1),
                                               self._ds,
                                               axis=2)
                    x[1, ...] += self._free_short_frac * self.safe_decay( self._tr[it] + self._te1, self._T2star_free_short) \
                                    * self._xp.conj(self.sens[i_sens]) * tmp
                    # second echo - bound compartment - long
                    tmp = self._xp.zeros(y[i_sens, 1, ...].shape,
                                         dtype=y.dtype)
                    tmp[self.readout_inds[it]] = y[i_sens, 1,
                                                   ...][self.readout_inds[it]]
                    tmp = self._xp.fft.ifftn(tmp, norm='ortho')
                    tmp = downsample_transpose(downsample_transpose(downsample_transpose(tmp, self._ds, axis=0),
                                                                    self._ds,
                                                                    axis=1),
                                               self._ds,
                                               axis=2)
                    x[1, ...] += self._free_short_frac * self.safe_decay( self._tr[it] + self._te1 + self._dt, self._T2star_free_short) \
                                    * self._xp.conj(self.sens[i_sens]) * tmp

        # get x gam back from GPU
        if self._xp.__name__ == 'cupy':
            x = self._xp.asnumpy(x)
        #----------------------

        # convert complex128 arrays back to 2 float64 array
        x = real_view_of_complex_array(x)

        return x


    #------------------------------------------------------------------------------

    def gradient(self, y: np.ndarray, var_dict: dict[VarName,Var], var_name: VarName) -> np.ndarray:
        return self.adjoint(y, var_dict)



class MonoExpDualTESodiumAcqModel(DualTESodiumAcqModel):
    """ mono exponential dual TE sodium acquisition model assuming one compartment """

    def __init__(self,
                 ds: int,
                 sens: XpArray,
                 dt: float,
                 te1: float,
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
        te1 : float
            TE1, start of the first acquisition
        readout_time : typing.Callable[[np.ndarray], np.ndarray]
            Callable that maps 1d kspace array to readout time for used read out
        kspace_part : RadialKSpacePartitioner
            RadialKSpacePartioner that partitions cartesian k-space volume
            into radial shells of "same" readout time
        """
        super().__init__(ds,
                         sens,
                         dt,
                         te1,
                         readout_time,
                         kspace_part)

    #------------------------------------------------------------------------------
    def forward(self, var_dict: dict[VarName,Var]) -> np.ndarray:
        """ Calculate apodized FFT of an image f

            Parameters
            ----------

            u : dict[VarName,Var]
                the list of image space variables that represent known or unknown parameters of the forward model,
                -> here two variables:
                   1) the image "proportional" to Na concentration, with shape (spatial dimensions, 2 - real and imaginary dimensions)
                   2) the Gamma, monoexponential T2* decay from TE1 to TE2, with shape (spatial dimensions)

            Returns
            -------
            a float64 numpy array of shape (self.num_coils, spatial dimensions, 2 - real and imaginary dimensions)
        """
        # read the input variables and create a complex view if required
        x = complex_view_of_real_array(var_dict[VarName.IMAGE].value)
        gam = var_dict[VarName.GAMMA].value

        #----------------------
        # send x and gam to GPU
        if self._xp.__name__ == 'cupy':
            x = self._xp.asarray(x)
            gam = self._xp.asarray(gam)

        y = self._xp.zeros(self.y_shape_complex, dtype=self._xp.complex128)


        for i_sens in range(self._num_coils):
            for it in range(self.n_readout_bins):
                temp = self.sens[i_sens, ...] * gam**((self._tr[it] + self._te1) / self._dt) * x
                temp =  downsample(downsample(downsample(temp, self._ds, axis=0),
                                              self._ds,
                                              axis=1),
                                   self._ds,
                                   axis=2)

                y[i_sens, 0, ...][self.readout_inds[it]] = self._xp.fft.fftn(temp, norm='ortho')[self.readout_inds[it]]

                temp = self.sens[i_sens, ...] * gam**(((self._tr[it] + self._te1)/ self._dt) + 1) * x
                temp =  downsample(downsample(downsample(temp, self._ds, axis=0),
                                              self._ds,
                                              axis=1),
                                   self._ds,
                                   axis=2)

                y[i_sens, 1, ...][self.readout_inds[it]] = self._xp.fft.fftn(temp, norm='ortho')[self.readout_inds[it]]

        # get x from GPU
        if self._xp.__name__ == 'cupy':
            y = self._xp.asnumpy(y)
        #----------------------

        # convert complex128 arrays back to 2 float64 array
        y = real_view_of_complex_array(y)

        return y

    #------------------------------------------------------------------------------
    def adjoint(self, y: np.ndarray, var_dict: dict[VarName,Var]) -> np.ndarray:
        """ Calculate the adjoint of the apodized FFT of a k-space image F

            Parameters
            ----------

            y : array of data shape

            u : dict[VarName,Var]
                the list of image space variables that represent known or unknown parameters of the forward model,
                -> here two variables:
                   1) the image "proportional" to Na concentration, with shape (spatial dimensions, 2 - real and imaginary dimensions)
                   2) the Gamma, monoexponential T2* decay from TE1 to TE2, with shape (spatial dimensions)

            Returns
            -------
            a float64 numpy array of shape (spatial dimensions, 2 - real and imaginary dimensions)
        """
        # create a complex view of the input real input array with two channels
        y = complex_view_of_real_array(y)

        gam = var_dict[VarName.GAMMA].value

        #----------------------
        # send y, gam to GPU
        if self._xp.__name__ == 'cupy':
            y = self._xp.asarray(y)
            gam = self._xp.asarray(gam)

        x = self._xp.zeros(var_dict[VarName.IMAGE].complex_shape,
                              dtype=self._xp.complex128)

        for i_sens in range(self._num_coils):
            for it in range(self.n_readout_bins):
                # 1st echo
                tmp0 = self._xp.zeros(y[i_sens, 0, ...].shape, dtype=y.dtype)
                tmp0[self.readout_inds[it]] = y[i_sens, 0,
                                                ...][self.readout_inds[it]]
                tmp0 = self._xp.fft.ifftn(tmp0, norm='ortho')

                tmp0 = downsample_transpose(downsample_transpose(downsample_transpose(tmp0, self._ds, axis=0),
                                               self._ds,
                                               axis=1),
                                    self._ds,
                                    axis=2)


                x += (gam**((self._tr[it] + self._te1) / self._dt)) * self._xp.conj(self.sens[i_sens]) * tmp0

                # 2nd echo
                tmp1 = self._xp.zeros(y[i_sens, 1, ...].shape, dtype=y.dtype)
                tmp1[self.readout_inds[it]] = y[i_sens, 1,
                                                ...][self.readout_inds[it]]

                tmp1 = self._xp.fft.ifftn(tmp1, norm='ortho')

                tmp1 = downsample_transpose(downsample_transpose(downsample_transpose(tmp1, self._ds, axis=0),
                                               self._ds,
                                               axis=1),
                                    self._ds,
                                    axis=2)

                x += (gam**(((self._tr[it] + self._te1)/ self._dt) + 1)) * self._xp.conj(self.sens[i_sens]) * tmp1

        # get x gam back from GPU
        if self._xp.__name__ == 'cupy':
            x = self._xp.asnumpy(x)
        #----------------------

        # convert complex128 arrays back to 2 float64 array
        x = real_view_of_complex_array(x)

        return x

    #------------------------------------------------------------------------------

    def gradient(self, y: np.ndarray, var_dict: dict[VarName,Var], var_name: VarName) -> np.ndarray:
        """ Calculate the "inner" derivative with respect to the first variable in the list u (here gamma)

            Parameters
            ----------

            y : a float64 numpy array of shape (self.num_coils, data_shape, 2 - real and imaginary dimensions)
                containing the "outer derivative" of the cost function

            u : dict[VarName,Var]
                the list of image space variables that represent known or unknown parameters of the forward model,
                -> here two variables:
                1) the gamma, monoexponential T2* decay from TE1 to TE2, real
                2) the image proportional to Na concentration, with shape (spatial dimensions, 2 - real and imaginary parts)

            Returns
            -------
            a float64 numpy array of shape (image spatial dimensions)
        """

        if var_name == VarName.IMAGE:
            return self.adjoint(y, var_dict)

        # create a complex view of the input real input array with two channels
        y = complex_view_of_real_array(y)

        img = complex_view_of_real_array(var_dict[VarName.IMAGE].value)
        gam = var_dict[VarName.GAMMA].value

        #----------------------
        # send y, gam to GPU
        if self._xp.__name__ == 'cupy':
            y = self._xp.asarray(y)
            gam = self._xp.asarray(gam)
            img = self._xp.asarray(img)

        x = self._xp.zeros(var_dict[VarName.IMAGE].complex_shape,
                              dtype=self._xp.complex128)

        for i_sens in range(self._num_coils):
            for it in range(self.n_readout_bins):
                n = (self._tr[it] + self._te1) / self._dt

                # 1st echo
                tmp0 = self._xp.zeros(y[i_sens, 0, ...].shape, dtype=y.dtype)
                tmp0[self.readout_inds[it]] = y[i_sens, 0,
                                                ...][self.readout_inds[it]]
                tmp0 = self._xp.fft.ifftn(tmp0, norm='ortho')

                tmp0 = downsample_transpose(downsample_transpose(downsample_transpose(tmp0, self._ds, axis=0),
                                               self._ds,
                                               axis=1),
                                    self._ds,
                                    axis=2)


                x += n * (gam**(n - 1)) * self._xp.conj(
                    img.squeeze() * self.sens[i_sens]) * tmp0

                # 2nd echo
                tmp1 = self._xp.zeros(y[i_sens, 1, ...].shape, dtype=y.dtype)
                tmp1[self.readout_inds[it]] = y[i_sens, 1,
                                                ...][self.readout_inds[it]]
                tmp1 = self._xp.fft.ifftn(tmp1, norm='ortho')

                tmp1 = downsample_transpose(downsample_transpose(downsample_transpose(tmp1, self._ds, axis=0),
                                               self._ds,
                                               axis=1),
                                    self._ds,
                                    axis=2)
                x += (n + 1) * (gam**n) * self._xp.conj(
                    img.squeeze() * self.sens[i_sens]) * tmp1

        # get x gam back from GPU
        if self._xp.__name__ == 'cupy':
            x = self._xp.asnumpy(x)
        #----------------------

        # convert complex128 arrays back to 2 float64 array
        x = real_view_of_complex_array(x)

        return x[..., 0]

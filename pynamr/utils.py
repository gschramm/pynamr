"""utils for pynamr models"""
import typing
import abc
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pymirc.viewer as pv
from scipy.ndimage import zoom

# check whether cupy is available
try:
    import cupy as cp
except ModuleNotFoundError:
    import numpy as cp

XpArray = typing.Union[np.ndarray, cp.ndarray]

  
class RadialKSpacePartitioner:
    """partion cartesion volume of kspace points into a number of shells"""

    def __init__(self,
                 data_shape: tuple[int, int, int],
                 n_readout_bins: int,
                 k_edge: float = 1.8 * 0.8197) -> None:

        self._data_shape = data_shape
        self._n_readout_bins = n_readout_bins
        self._k_edge = k_edge

        # setup the readout time and readout indexes
        k0, k1, k2 = np.meshgrid(np.linspace(-k_edge, k_edge, data_shape[0]),
                                 np.linspace(-k_edge, k_edge, data_shape[1]),
                                 np.linspace(-k_edge, k_edge, data_shape[2]))

        abs_k = np.sqrt(k0**2 + k1**2 + k2**2)
        abs_k = np.fft.fftshift(abs_k)

        k_1d = np.linspace(0, k_edge, n_readout_bins + 1)

        self._readout_inds = []
        self._kmask = np.zeros(data_shape, dtype=np.uint8)
        self._k = np.zeros(n_readout_bins)

        for i in range(n_readout_bins):
            rinds = np.where(
                np.logical_and(abs_k >= k_1d[i], abs_k < k_1d[i + 1]))
            self._readout_inds.append(rinds)
            self.kmask[rinds] = 1
            self._k[i] = 0.5 * (k_1d[i] + k_1d[i + 1])

        # convert mutable list to tuple
        self._readout_inds = tuple(self._readout_inds)
        # convert kmask mask into a pseudo "complex" array
        self._kmask = np.stack([self._kmask, self._kmask], axis=-1)

    @property
    def kmask(self) -> np.ndarray:
        return self._kmask

    @property
    def k(self) -> np.ndarray:
        return self._k

    @property
    def k_edge(self) -> float:
        return self._k_edge

    @property
    def n_readout_bins(self) -> int:
        return self._n_readout_bins

    @property
    def readout_inds(self) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return self._readout_inds

    def show(self,
             fftshift: bool = True,
             sl: typing.Optional[int] = None,
             **kwargs) -> None:
        vol = np.zeros(self._data_shape, dtype=np.int16)

        for i, inds in enumerate(self._readout_inds):
            vol[inds] = i + 1

        if fftshift:
            vol = np.fft.fftshift(vol)

        if sl is None:
            sl = self._data_shape[2] // 2

        fig, ax = plt.subplots()
        ax.imshow(vol[:, :, sl], **kwargs)
        fig.tight_layout()
        fig.show()


class RadialKSpaceReadOutTime(abc.ABC):
    """ABC for mapping of absolute k-space values to readout times"""

    def __call__(self, k: XpArray) -> XpArray:
        raise NotImplementedError


class TPIReadOutTime(RadialKSpaceReadOutTime):
    """mapping of absolute k-space values to readout time (ms) for TPI sequence"""

    def __init__(self,
                 eta: float = 0.9830,
                 c1: float = 0.54,
                 c2: float = 0.46,
                 alpha_sw_tpi: float = 18.95,
                 beta_sw_tpi: float = -0.5171,
                 t0_sw: float = 0.0018) -> None:
        self._eta = eta
        self._c1 = c1
        self._c2 = c2
        self._alpha_sw_tpi = alpha_sw_tpi
        self._beta_sw_tpi = beta_sw_tpi
        self._t0_sw = t0_sw

    def __call__(self, k: XpArray) -> XpArray:
        if isinstance(k, np.ndarray):
            xp = np
        else:
            xp = cp

        # the point until the readout is linear
        m = 1126 * 0.16

        k_lin = self._t0_sw * m

        i1 = xp.where(k <= k_lin)
        i2 = xp.where(k > k_lin)

        t = xp.zeros(k.shape)

        t[i1] = k[i1] / m
        t[i2] = self._t0_sw + ((self._c2 * (k[i2]**3) - (
            (self._c1 / self._eta) * np.exp(-self._eta *
                                            (k[i2]**3))) - self._beta_sw_tpi) /
                               (3 * self._alpha_sw_tpi))

        # convert to ms
        t *= 1000

        return t

class TPIInstantaneousReadOutTime(RadialKSpaceReadOutTime):
    """mapping of absolute k-space values to zero (instanteneous) readout time (ms) for TPI trajectory"""

    def __init__(self) -> None:
        pass
    def __call__(self, k: XpArray) -> XpArray:
        t = np.zeros(k.shape)
        return t

def complex_view_of_real_array(x: np.ndarray) -> np.ndarray:
    """return a complex view of of a real array, by interpreting the last dimension as as real and imaginary part
       output = input[...,0] + 1j * input[....,1]
    """
    if x.dtype == np.float64:
        return np.squeeze(x.view(dtype=np.complex128), axis=-1)
    elif x.dtype == np.float32:
        return np.squeeze(x.view(dtype=np.complex64), axis=-1)
    elif x.dtype == np.float128:
        return np.squeeze(x.view(dtype=np.complex256), axis=-1)
    else:
        raise ValueError('Input must have dtyoe float32, float64 or float128')


def real_view_of_complex_array(x: np.ndarray) -> np.ndarray:
    """return a real view of a complex array
       output[...,0] = real(input)
       output[...,1] = imaginary(input)
    """
    return np.stack([x.real, x.imag], axis=-1)


def downsample(x: XpArray, ds: int, axis: int = 0) -> XpArray:
    """downsample a numpy/cupy array along a given axis by an integer factor
    
       Parameters
       ----------
       x : the array to downsample

       ds : the downsampling factor

       axis : axis along which to downsample

       Returns
       -------
       Downsampled XpArray
    """

    # test whether the input is a numpy or cupy array
    if isinstance(x, np.ndarray):
        xp = np
    else:
        xp = cp

    ds_shape = list(x.shape)
    ds_shape[axis] = ds_shape[axis] // ds
    ds_shape = tuple(ds_shape)

    x_ds = xp.zeros(ds_shape, dtype=x.dtype)

    sl = slice(None, None, None)

    for i in range(ds_shape[axis]):
        sl1 = x.ndim * [sl]
        sl1[axis] = slice(ds * i, ds * (i + 1), None)
        sl1 = tuple(sl1)

        sl2 = x.ndim * [sl]
        sl2[axis] = slice(i, i + 1, None)
        sl2 = tuple(sl2)

        x_ds[sl2] = x[sl1].sum(axis=axis, keepdims=True)

    return x_ds / ds


def downsample_transpose(x_ds: XpArray, ds: int, axis: int = 0) -> XpArray:
    """transpose of downsample of a numpy/cupy array along a given axis by an integer factor
    
       Parameters
       ----------
       x : the input array

       ds : the upsampling factor

       axis : axis along which to apply the operator

       Returns
       -------
       Upsampled XpArray
    """

    # test whether the input is a numpy or cupy array
    if isinstance(x_ds, np.ndarray):
        xp = np
    else:
        xp = cp

    up_shape = list(x_ds.shape)
    up_shape[axis] = up_shape[axis] * ds
    up_shape = tuple(up_shape)

    x = xp.zeros(up_shape, dtype=x_ds.dtype)

    sl = slice(None, None, None)

    for i in range(x_ds.shape[axis]):
        sl1 = x_ds.ndim * [sl]
        sl1[axis] = slice(ds * i, ds * (i + 1), None)
        sl1 = tuple(sl1)

        sl2 = x_ds.ndim * [sl]
        sl2[axis] = slice(i, i + 1, None)
        sl2 = tuple(sl2)

        x[sl1] = x_ds[sl2]

    return x / ds

def upsample_nearest(x_ds: XpArray, ds: int, axis: int = 0) -> XpArray:
    """upsample a numpy/cupy array along a given axis by an integer factor
       using a nearest neighbour interpolation
    
       Parameters
       ----------
       x : the array to upsample

       ds : the upsampling factor

       axis : axis along which to upsample

       Returns
       -------
       Upsampled XpArray
    """

    # test whether the input is a numpy or cupy array
    if isinstance(x_ds, np.ndarray):
        xp = np
    else:
        xp = cp

    up_shape = list(x_ds.shape)
    up_shape[axis] = up_shape[axis] * ds
    up_shape = tuple(up_shape)

    x = xp.zeros(up_shape, dtype=x_ds.dtype)

    sl = slice(None, None, None)

    for i in range(x_ds.shape[axis]):
        sl1 = x_ds.ndim * [sl]
        sl1[axis] = slice(ds * i, ds * (i + 1), None)
        sl1 = tuple(sl1)

        sl2 = x_ds.ndim * [sl]
        sl2[axis] = slice(i, i + 1, None)
        sl2 = tuple(sl2)

        x[sl1] = x_ds[sl2]

    return x 


def sum_of_squares_reconstruction(data: np.ndarray, complex_format = False) -> np.ndarray:
    """ sum of squares reconstruction of multi-channel data

    Parameters
    ----------
    data : np.ndarray
        Real array of shape (ncoils, n0, n1, n2, 2) containing the multi-channel
        coil data. The last dimension is used to store real and imaginary part.

    Returns
    -------
    np.ndarray
        Real array of shape (n0,n1,n2) with the sum of squares reconstruction
    """    
    tmp = []
    for icoil in range(data.shape[0]):
        data_tmp = (data[icoil,...] if complex_format else complex_view_of_real_array(data[icoil,...]))
        tmp.append(np.fft.ifftn(data_tmp, norm='ortho'))
    
    tmp = np.array(tmp)
    
    sos = np.sqrt((np.abs(tmp)**2).sum(axis=0))

    return sos
 
def simple_reconstruction(data: np.ndarray, complex_format = False) -> np.ndarray:
    """ the simplest recon possible from single-channel data (IFFT)

    Parameters
    ----------
    data : np.ndarray
        Real array of shape (n0, n1, n2, 2) containing the single-channel
        coil data. The last dimension is used to store real and imaginary part.

    Returns
    -------
    np.ndarray
        Real array of shape (n0, n1, n2, 2) with the simple reconstruction.
        The last dimension is used to store real and imaginary part.
    """    
    data_tmp = (data if complex_format else complex_view_of_real_array(data))
    recon = np.fft.ifftn(data_tmp, norm='ortho')

    return recon

# simple ITK conversion
def numpy_volume_to_sitk_image (vol: np.ndarray, voxel_size: np.ndarray, origin: np.ndarray) -> np.ndarray:
  image = sitk.GetImageFromArray(np.swapaxes(vol, 0, 2))
  image.SetSpacing(voxel_size.astype(np.float64))
  image.SetOrigin(origin.astype(np.float64))

  return image

# simple ITK conversion
def sitk_image_to_numpy_volume (img: np.ndarray) -> np.ndarray:
  vol = np.swapaxes(sitk.GetArrayFromImage(img), 0, 2)

  return vol


# Registration of higher resolution proton MRI to Na MRI image using simple ITK
def register_highresH_to_lowresNa (highresH: np.ndarray, lowresNa: np.ndarray, highresH_voxsize: np.ndarray, lowresNa_voxsize: np.ndarray, highresH_origin: np.ndarray, lowresNa_origin: np.ndarray) -> np.ndarray:

    moving_image  = numpy_volume_to_sitk_image(highresH.astype(np.float32), highresH_voxsize, highresH_origin)
    fixed_image = numpy_volume_to_sitk_image(lowresNa.astype(np.float32), lowresNa_voxsize, lowresNa_origin)

    # Initial Alignment
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image,
                                                      sitk.Euler3DTransform(),
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)

    # Registration
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins = 50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)

    registration_method.SetInterpolator(sitk.sitkLinear)


    registration_method.SetOptimizerAsGradientDescentLineSearch(
        learningRate            = 0.3,
        numberOfIterations      = 200,
        convergenceMinimumValue = 1e-7,
        convergenceWindowSize   = 10)

    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace = False)

    final_transform = registration_method.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32))

    # Post registration analysis
    print(f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}")
    print(f"Final metric value: {registration_method.GetMetricValue()}")
    print(f"Final parameters: {final_transform.GetParameters()}")

    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                     moving_image.GetPixelID())

    highresH_img_aligned = sitk_image_to_numpy_volume(moving_resampled)

    return highresH_img_aligned

# N4 correction for smooth inhomogeneities in MRI
def n4 (img: np.ndarray) -> np.ndarray:
    image = sitk.GetImageFromArray(np.swapaxes(img, 0, 2))

    maskImage = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    numberFittingLevels = 4

    corrected_image = corrector.Execute(image, maskImage)
    corrected_image = np.swapaxes(sitk.GetArrayFromImage(corrected_image),0,2)

    return corrected_image

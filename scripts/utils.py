import math
import numpy as np
import numpy.typing as npt
from numba import jit


def tpi_sampling_density(kx: npt.NDArray, ky: npt.NDArray, kz: npt.NDArray,
                         kp: float):
    kabs = np.sqrt(kx**2 + ky**2 + kz**2)

    sampling_density = (kabs**2)
    sampling_density[kabs > kp] = kp**2

    return sampling_density


@jit(nopython=True)
def trilinear_kspace_interpolation(non_uniform_data, kx, ky, kz, matrix_size,
                                   delta_k, kmax, output):

    for i in range(kx.size):
        kabs = np.sqrt(kx[i]**2 + ky[i]**2 + kz[i]**2)
        if kabs <= kmax:

            kx_shifted = (kx[i] / delta_k) + 0.5 * (matrix_size)
            ky_shifted = (ky[i] / delta_k) + 0.5 * (matrix_size)
            kz_shifted = (kz[i] / delta_k) + 0.5 * (matrix_size)

            kx_shifted_low = math.floor(kx_shifted)
            ky_shifted_low = math.floor(ky_shifted)
            kz_shifted_low = math.floor(kz_shifted)

            kx_shifted_high = kx_shifted_low + 1
            ky_shifted_high = ky_shifted_low + 1
            kz_shifted_high = kz_shifted_low + 1

            dkx = float(kx_shifted - kx_shifted_low)
            dky = float(ky_shifted - ky_shifted_low)
            dkz = float(kz_shifted - kz_shifted_low)

            toAdd = non_uniform_data[i]

            if (kx_shifted_low >= 0) and (ky_shifted_low >=
                                          0) and (kz_shifted_low >= 0):

                output[kx_shifted_low, ky_shifted_low,
                       kz_shifted_low] += (1 - dkx) * (1 - dky) * (1 -
                                                                   dkz) * toAdd

                output[kx_shifted_high, ky_shifted_low,
                       kz_shifted_low] += (dkx) * (1 - dky) * (1 - dkz) * toAdd

                output[kx_shifted_low, ky_shifted_high,
                       kz_shifted_low] += (1 - dkx) * (dky) * (1 - dkz) * toAdd

                output[kx_shifted_high, ky_shifted_high,
                       kz_shifted_low] += (dkx) * (dky) * (1 - dkz) * toAdd

                output[kx_shifted_low, ky_shifted_low,
                       kz_shifted_high] += (1 - dkx) * (1 -
                                                        dky) * (dkz) * toAdd

                output[kx_shifted_high, ky_shifted_low,
                       kz_shifted_high] += (dkx) * (1 - dky) * (dkz) * toAdd

                output[kx_shifted_low, ky_shifted_high,
                       kz_shifted_high] += (1 - dkx) * (dky) * (dkz) * toAdd

                output[kx_shifted_high, ky_shifted_high,
                       kz_shifted_high] += (dkx) * (dky) * (dkz) * toAdd


class TriliniearKSpaceRegridder:

    def __init__(self,
                 matrix_size: int,
                 delta_k: float,
                 kx: npt.NDArray,
                 ky: npt.NDArray,
                 kz: npt.NDArray,
                 sampling_density: npt.NDArray,
                 kmax: float | None = None,
                 phase_correct: bool = True,
                 normalize_central_weight=True) -> None:
        self._matrix_size = matrix_size
        self._delta_k = delta_k
        self._phase_correct = phase_correct
        self._normalize_central_weight = normalize_central_weight

        self._kx = kx
        self._ky = ky
        self._kz = kz

        self._kabs = np.sqrt(kx**2 + ky**2 + kz**2)

        self._sampling_density = sampling_density.astype(np.float64)

        if kmax is None:
            self._kmax = np.max(self.kabs)
        else:
            self._kmax = kmax

        self._sampling_weights = np.zeros(
            (self.matrix_size, self.matrix_size, self.matrix_size),
            dtype=np.float64)

        # calculate the sampling weights which we need to correct for the
        # if we want to correct for the sampling density in the center
        trilinear_kspace_interpolation(self._sampling_density, self._kx,
                                       self._ky, self._kz, self._matrix_size,
                                       self._delta_k, self._kmax,
                                       self._sampling_weights)

        self._central_weight = self._sampling_weights[self._matrix_size // 2,
                                                      self._matrix_size // 2,
                                                      self._matrix_size // 2]

        if self._normalize_central_weight:
            self._sampling_weights /= self._central_weight

        # create the phase correction field containing a checkerboard pattern
        tmp_x = np.arange(self._matrix_size)
        TMP_X, TMP_Y, TMP_Z = np.meshgrid(tmp_x, tmp_x, tmp_x)

        self._phase_correction_field = ((-1)**TMP_X) * ((-1)**TMP_Y) * (
            (-1)**TMP_Z)

        # create the correction field due to interpolation (FT of interpolation kernel)
        tmp_x = np.linspace(-0.5, 0.5, self._matrix_size)
        TMP_X, TMP_Y, TMP_Z = np.meshgrid(tmp_x, tmp_x, tmp_x)
        self._interpolation_correction_field = np.sinc(
            np.sqrt(TMP_X**2 + TMP_Y**2 + TMP_Z**2))**2

    @property
    def kabs(self) -> npt.NDArray:
        return self._kabs

    @property
    def matrix_size(self) -> int:
        return self._matrix_size

    @property
    def delta_k(self) -> float:
        return self._delta_k

    @property
    def kx(self) -> npt.NDArray:
        return self._kx

    @property
    def ky(self) -> npt.NDArray:
        return self._ky

    @property
    def kz(self) -> npt.NDArray:
        return self._kz

    @property
    def sampling_density(self) -> npt.NDArray:
        return self._sampling_density

    @property
    def kmax(self) -> float:
        return self._kmax

    @property
    def sampling_weights(self) -> npt.NDArray:
        return self._sampling_weights

    @property
    def central_weight(self) -> float:
        return self._central_weight

    @property
    def phase_correct(self) -> bool:
        return self._phase_correct

    @property
    def phase_correction_field(self) -> npt.NDArray:
        return self._phase_correction_field

    @property
    def interpolation_correction_field(self) -> npt.NDArray:
        return self._interpolation_correction_field

    def regrid(self, non_uniform_data: npt.NDArray) -> npt.NDArray:
        output = np.zeros(
            (self.matrix_size, self.matrix_size, self.matrix_size),
            dtype=non_uniform_data.dtype)

        trilinear_kspace_interpolation(
            self._sampling_density * non_uniform_data, self._kx, self._ky,
            self._kz, self._matrix_size, self._delta_k, self._kmax, output)

        output = np.fft.fftshift(output)

        if self._phase_correct:
            output *= self._phase_correction_field

        if self._normalize_central_weight:
            output /= self._central_weight

        # correct for matrix size, important when we use ffts with norm = 'ortho'
        output /= np.sqrt((128 / self._matrix_size)**3)

        # correct for fall-off towards the edges due to interpolation
        ifft_output = np.fft.ifftn(
            output, norm='ortho') / self._interpolation_correction_field
        output = np.fft.fftn(ifft_output, norm='ortho')

        return output

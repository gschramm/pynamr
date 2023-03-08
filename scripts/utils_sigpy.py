import sigpy
import math
import numpy as np
import cupy as cp

from typing import Union


def nufft_t2star_operator(
    ishape: tuple[int, int, int],
    k: np.ndarray,
    field_of_view_cm: float = 22.,
    acq_sampling_time_ms: float = 0.01,
    time_bin_width_ms: float = 0.25,
    scale: float = 0.03,
    add_mirrored_coordinates=True,
    ratio_image: Union[cp.ndarray, None] = None,
    echo_time_1_ms: float = 0.5,
    echo_time_2_ms: float = 5.
) -> Union[sigpy.linop.Linop, Union[sigpy.linop.Linop, sigpy.linop.Linop]]:
    """sigpy (dual echo) forward nufft operator including monoexp. T2* decay modeling

    Parameters
    ----------
    ishape : tuple[int, int, int]
        shape of the input image
    k : np.ndarray
        input kx, ky, kz coordinates - shape: (num_samples, num_readouts, 3)
        units 1/cm
    field_of_view_cm : float, optional
        field of view in cm, by default 220.
    acq_sampling_time_ms : float, optional
        samplignt time during acquisition in ms, by default 0.01
    time_bin_width_ms : float, optional
        time bin width for modeling T2* decay, by default 0.25
    scale : float, optional
        scale of the forward operator, by default 0.03
    add_mirrored_coordinates : bool, optional
        add -kx, -ky, -kz to coordinates, by default True
    ratio_image : Union[cp.ndarray, None], optional
        ratio between first and second echo image for modeling T2* decay, by default None
        None means that no decay is modeled
    echo_time_1_ms : float, optional
        first echo time in ms, by default 0.5
    echo_time_2_ms : float, optional
        second echo time in ms, by default 5.

    Returns
    -------
    Union[sigpy.linop.LinOp, Union[sigpy.linop.LinOp, sigpy.linop.LinOp]]
        single sigpy linear operator if ratio image is None
        two sigpy linear operators if ratio image is not None
    """
    time_bins_inds = np.array_split(
        np.arange(k.shape[0]),
        math.ceil(k.shape[0] / (time_bin_width_ms / acq_sampling_time_ms)))

    # split the k-space coordinates into time bins
    all_coords = []

    for _, time_bin_inds in enumerate(time_bins_inds):
        chunk_coords_1_cm = k[time_bin_inds, :, :].reshape(-1, 3)

        # the gradient files only contain a half sphere
        # we add the 2nd half where all gradients are reversed
        if add_mirrored_coordinates:
            chunk_coords_1_cm = cp.vstack(
                (chunk_coords_1_cm, -chunk_coords_1_cm))

        all_coords.append(chunk_coords_1_cm * field_of_view_cm)

    if ratio_image is None:
        operator1 = scale * sigpy.linop.NUFFT(
            ishape, cp.vstack(all_coords), oversamp=1.25, width=4)
        return operator1
    else:
        op1s = []
        op2s = []
        for i, time_bin_inds in enumerate(time_bins_inds):
            #setup the decay image
            readout_time_1_ms = echo_time_1_ms + time_bin_inds.mean(
            ) * acq_sampling_time_ms
            readout_time_2_ms = echo_time_2_ms + time_bin_inds.mean(
            ) * acq_sampling_time_ms

            decay_image_1 = ratio_image**((readout_time_1_ms) /
                                          (echo_time_2_ms - echo_time_1_ms))
            decay_image_2 = ratio_image**((readout_time_2_ms) /
                                          (echo_time_2_ms - echo_time_1_ms))

            op1s.append(scale * sigpy.linop.NUFFT(
                ishape, all_coords[i], oversamp=1.25, width=4) *
                        sigpy.linop.Multiply(ishape, decay_image_1))
            op2s.append(scale * sigpy.linop.NUFFT(
                ishape, all_coords[i], oversamp=1.25, width=4) *
                        sigpy.linop.Multiply(ishape, decay_image_2))

        # operator including T2* decay for first echo and second echo
        operator1 = sigpy.linop.Vstack(op1s)
        operator2 = sigpy.linop.Vstack(op2s)

        return operator1, operator2

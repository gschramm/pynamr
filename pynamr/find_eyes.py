import argparse
from pathlib import Path
import math

import numpy as np
import nibabel as nib

import matplotlib.pyplot as plt

import scipy.ndimage as sndi


def load_nii(fname):
    nii = nib.load(fname)
    nii = nib.as_closest_canonical(nii)
    vol = nii.get_fdata()

    return vol, nii.header


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fname', help='nifti input Na file')
    parser.add_argument('--kernelsize_mm',
                        type=float,
                        default=20,
                        help='eye kernel size (mm)')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    fname: str = args.fname
    kernelsize_mm: float = args.kernelsize_mm
    show: bool = args.show

    # load the data
    vol, hdr = load_nii(fname)
    pixsize = hdr["pixdim"][1]

    # downsample volume to lower resolution to get faster processing
    downsample_factor = math.ceil(2 / pixsize)

    if downsample_factor > 1:
        vol = vol[::downsample_factor, ::downsample_factor, ::
                  downsample_factor]
        pixsize *= downsample_factor

    kernelsize = int(kernelsize_mm / pixsize)
    th = np.percentile(vol, 99.5)

    # generate a sphere like kernel that we use to enhance spherical objects of 2cm diameter
    nos = 5
    x = np.linspace(-4.5, 4.5, nos * kernelsize)
    X, Y, Z = np.meshgrid(x, x, x)
    M = sndi.gaussian_filter(
        (np.sqrt(X**2 + Y**2 + Z**2) < 4.5).astype(np.float64), 5)
    M = M[(nos // 2)::nos, (nos // 2)::nos, (nos // 2)::nos]
    eye_kernel = 2 * M - 1

    eye_mask = sndi.convolve((vol > th).astype(np.float64), eye_kernel)
    eye_mask2 = eye_mask > 0.6 * eye_mask.max()
    eye_mask2[:, :(eye_mask.shape[1] // 2), :] = 0

    # calculate the center of mass of all labels to keep the ones that have the
    # smallest AP and FH distance

    label_img, n_labels = sndi.label(eye_mask2)

    coms = np.zeros((n_labels, 3))

    for i, lab in enumerate(range(1, n_labels + 1)):
        coms[i, :] = sndi.center_of_mass(label_img == lab)

    min_dist = np.finfo(np.float64).max

    for i in range(n_labels):
        for j in range(i + 1, n_labels):
            dist = (coms[i][1] - coms[j][1])**2 + (coms[i][2] - coms[j][2])**2

            if dist < min_dist:
                imin = i
                jmin = j

    eye_coords = np.array([
        sndi.center_of_mass(label_img == (imin + 1)),
        sndi.center_of_mass(label_img == (jmin + 1))
    ])

    # interpolate back to original voxel size
    if downsample_factor > 1:
        vol = sndi.zoom(vol, downsample_factor, order=1, prefilter=False)
        pixsize /= downsample_factor
        eye_coords *= downsample_factor

    # save the results
    np.savetxt(
        Path(fname).parent / f'{Path(fname).stem}_eye_coordinates.txt',
        eye_coords)

    # show results:
    if show:
        import pymirc.viewer as pv
        X0, X1, X2 = np.meshgrid(np.arange(vol.shape[0]),
                                 np.arange(vol.shape[1]),
                                 np.arange(vol.shape[2]),
                                 indexing='ij')

        R0 = np.sqrt((X0 - eye_coords[0, 0])**2 + (X1 - eye_coords[0, 1])**2 +
                     (X2 - eye_coords[0, 2])**2)
        R1 = np.sqrt((X0 - eye_coords[1, 0])**2 + (X1 - eye_coords[1, 1])**2 +
                     (X2 - eye_coords[1, 2])**2)

        vi = pv.ThreeAxisViewer(
            np.flip(vol, (0, 1)),
            np.flip((R1 < 6 / pixsize) + (R0 < 6 / pixsize), (0, 1)))
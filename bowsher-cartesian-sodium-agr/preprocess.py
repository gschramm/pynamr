"""preprocess sodium birdcage raw data for Bowsher AGR recon"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import nibabel as nib

import SimpleITK as sitk
import matplotlib.pyplot as plt
import pymirc.viewer as pv
import shutil
from scipy.ndimage import gaussian_filter, zoom

from utils import load_config, numpy_volume_to_sitk_image, sitk_image_to_numpy_volume

## %

parser = ArgumentParser()
parser.add_argument("--cfg", help="config json file", default="config.json")
parser.add_argument("--debug", action="store_true", help="debug mode")
args = parser.parse_args()

cfg = load_config(args.cfg)
debug: bool = args.debug

# create the output directory
odir = Path(cfg.recon_dir)
odir.mkdir(exist_ok=True)

# %%
# load complex Na images
# ----------------------

cimg = np.flip(
    np.fromfile(cfg.TE05c0_filename, dtype=np.complex64)
    .reshape(cfg.data_shape)
    .swapaxes(0, 2),
    (1, 2),
)

# phase rotate the complex image such that np.angle(cimg) = 0
cimg = cimg * np.exp(-1j * np.angle(cimg))


# perform fft to get into k-space
cimg_fft = np.fft.fftn(cimg, norm="ortho")
# pad data with 0s
data = np.fft.fftshift(
    np.pad(np.fft.fftshift(cimg_fft), (cfg.recon_shape[0] - cfg.data_shape[0]) // 2)
)

# load the complex coil image for 2nd echo
cimg2 = np.flip(
    np.fromfile(cfg.TE5c0_filename, dtype=np.complex64)
    .reshape(cfg.data_shape)
    .swapaxes(0, 2),
    (1, 2),
)


# phase rotate the complex image such that np.angle(cimg2) = 0
cimg2 = cimg2 * np.exp(-1j * np.angle(cimg2))


# perform fft to get into k-space
cimg_fft2 = np.fft.fftn(cimg2, norm="ortho")
# pad data with 0s
data2 = np.fft.fftshift(
    np.pad(np.fft.fftshift(cimg_fft2), (cfg.recon_shape[0] - cfg.data_shape[0]) // 2)
)

# expand dims to make add coil channel dim
data = np.expand_dims(data, 0)
data2 = np.expand_dims(data2, 0)
sens = np.ones_like(data)

# %%
# save complex Na data
# --------------------

# save the data and the sensitities
np.save(odir / f"echo1_{cfg.recon_shape[0]}.npy", data)
np.save(odir / f"echo2_{cfg.recon_shape[0]}.npy", data2)
np.save(odir / f"sens_{cfg.recon_shape[0]}.npy", sens)

# %%
# load binary MR data
# -------------------


# test whether cfg.MPRAGE_filename exists and is a nifti file
if not Path(cfg.MPRAGE_filename).exists():
    raise FileNotFoundError(f"MPRAGE file {cfg.MPRAGE_filename} not found")

if cfg.MPRAGE_filename.endswith((".nii")):
    # copy cfg.MPRAGE_filename to recon_dir as T1_input.nii without loading it
    # might need to to some heuristic flipping here to match sodium orientation
    shutil.copy(cfg.MPRAGE_filename, str(Path(cfg.recon_dir) / "T1_input.nii"))
    mr = nib.as_closest_canonical(nib.load(cfg.MPRAGE_filename)).get_fdata()
else:
    mr = np.flip(
        np.swapaxes(
            np.fromfile(cfg.MPRAGE_filename, dtype=np.uint16).reshape(192, 256, 256),
            0,
            2,
        ),
        (0, 1),
    )

    nib.save(
        nib.Nifti1Image(mr.astype(np.float32), np.eye(4)),
        str(Path(cfg.recon_dir) / "T1_input.nii"),
    )

# ---------------------------------------------------------------------------------------------------
# visualizations

a = np.abs(gaussian_filter(cimg, cfg.alignment_filter_sigma))
b = np.abs(gaussian_filter(cimg2, cfg.alignment_filter_sigma))

vmax = np.percentile(a, 99.99)

vi = pv.ThreeAxisViewer([a, b], imshow_kwargs={"vmax": vmax, "cmap": "Greys_r"})
vi.fig.savefig(Path(cfg.recon_dir) / "sodium_input.png")
vi2 = pv.ThreeAxisViewer(mr, imshow_kwargs={"cmap": "Greys_r"})
vi2.fig.savefig(Path(cfg.recon_dir) / "mr_input.png")

# %%
# N4 bias correction of the MPRAGE
# --------------------------------

n4_path = Path(cfg.recon_dir) / "T1_input_N4_corrected.nii"

print("running N4 bias correction on the MPRAGE")
inputImage = sitk.ReadImage(str(Path(cfg.recon_dir) / "T1_input.nii"), sitk.sitkFloat32)

corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrected_image = corrector.Execute(
    inputImage, sitk.OtsuThreshold(inputImage, 0, 1, 200)
)

sitk.WriteImage(corrected_image, str(n4_path))

# %%
# align the N4 corrected MPRAGE image
# -----------------------------------

aligned_path = Path(cfg.recon_dir) / f"T1_N4_corrected_aligned.npy"

print("running MR alignment")

# load the N4 corrected T1 in RAS
t1_nii = nib.load(str(Path(cfg.recon_dir) / "T1_input_N4_corrected.nii"))
t1_nii = nib.as_closest_canonical(t1_nii)
t1 = t1_nii.get_fdata()
t1_affine = t1_nii.affine
t1_voxsize = t1_nii.header["pixdim"][1:4]
t1_origin = t1_affine[:-1, -1]

na_img = np.abs(gaussian_filter(cimg, cfg.alignment_filter_sigma))
na_img2 = np.abs(gaussian_filter(cimg2, cfg.alignment_filter_sigma))

# interpolate to smaller grid
na_img = zoom(
    na_img,
    np.array(cfg.recon_shape) / np.array(cfg.data_shape),
    order=1,
    prefilter=False,
)

# construct the voxel size for the Na sos image
na_voxsize = cfg.sodium_fov_mm / np.array(cfg.recon_shape)
na_origin = np.full(3, -cfg.sodium_fov_mm / 2.0)

# %%
# SITK alignment using mututla information

fixed_image = numpy_volume_to_sitk_image(
    na_img.astype(np.float32), na_voxsize, na_origin
)
moving_image = numpy_volume_to_sitk_image(t1.astype(np.float32), t1_voxsize, t1_origin)

# Initial Alignment
initial_transform = sitk.CenteredTransformInitializer(
    fixed_image,
    moving_image,
    sitk.Euler3DTransform(),
    sitk.CenteredTransformInitializerFilter.GEOMETRY,
)

# Registration
registration_method = sitk.ImageRegistrationMethod()

# Similarity metric settings.
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.1)

registration_method.SetInterpolator(sitk.sitkLinear)

# Optimizer settings.
registration_method.SetOptimizerAsGradientDescentLineSearch(
    learningRate=0.3,
    numberOfIterations=200,
    convergenceMinimumValue=1e-7,
    convergenceWindowSize=10,
)

registration_method.SetOptimizerScalesFromPhysicalShift()

# Setup for the multi-resolution framework.
registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# Don't optimize in-place, we would possibly like to run this cell multiple times.
registration_method.SetInitialTransform(initial_transform, inPlace=False)

final_transform = registration_method.Execute(
    sitk.Cast(fixed_image, sitk.sitkFloat32),
    sitk.Cast(moving_image, sitk.sitkFloat32),
)

# Post registration analysis
print(
    f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}"
)
print(f"Final metric value: {registration_method.GetMetricValue()}")
print(f"Final parameters: {final_transform.GetParameters()}")

moving_resampled = sitk.Resample(
    moving_image,
    fixed_image,
    final_transform,
    sitk.sitkLinear,
    0.0,
    moving_image.GetPixelID(),
)

t1_aligned = sitk_image_to_numpy_volume(moving_resampled)

# save aligned T1
np.save(aligned_path, t1_aligned)

# %%
# show the aligned images

vi3 = pv.ThreeAxisViewer(
    [t1_aligned, na_img, t1_aligned],
    [None, None, na_img],
    imshow_kwargs=dict(cmap="Greys_r"),
)
vi3.fig.savefig(Path(cfg.recon_dir) / "mr_input_N4_corrected_aligned.png")

if debug:
    np.save(
        Path(cfg.recon_dir) / "t1_aligned_debug.npy",
        t1_aligned,
    )
    np.save(
        Path(cfg.recon_dir) / "na_img_debug.npy",
        na_img,
    )
    np.save(Path(cfg.recon_dir) / "na_img2_debug.npy", na_img2)

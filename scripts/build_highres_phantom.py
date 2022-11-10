import numpy as np
import nibabel as nib

from scipy.ndimage     import zoom, gaussian_filter
from scipy.interpolate import interp1d
from scipy.ndimage import binary_dilation, binary_fill_holes, binary_closing, grey_erosion, grey_closing
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import truncnorm

import matplotlib.pyplot as plt

from copy import deepcopy
from argparse import ArgumentParser
import sys
import os
from glob import glob
from datetime import datetime

import pymirc.viewer as pv
import pynamr


build_csf = False
compute_phantom = True
add_lesion = True

sdir = '/uz/data/Admin/ngeworkingresearch/MarinaFilipovic/Belzunce_Phantom'
odir = '/uz/data/Admin/ngeworkingresearch/MarinaFilipovic/Belzunce_Phantom/NaMRI' + ('_lesion' if add_lesion else '')
if not os.path.exists(odir):
    os.makedirs(odir)

# final cubic array size for the phantom
final_n = 128

seed = 42
np.random.seed(seed)

# brian tissue classes (gray matter, white matter, etc.) from 2 to 9, 1 is the background, no CSF
classes = nib.load(os.path.join(sdir, 'full_cls_400um_2009b_sym.nii'))
classes = nib.as_closest_canonical(classes)
classes_im = classes.get_fdata().astype(np.uint8)

# cell density from histology
cells = nib.load(os.path.join(sdir, 'full16_400um_2009b_sym.nii'))
cells = nib.as_closest_canonical(cells)
cells_im = cells.get_fdata()
cells_im = cells_im.max() - cells_im
cells_im /= cells_im.max()

# MRI
mri = nib.load(os.path.join(sdir,'mri_registered_2009b_sym.nii'))
mri = nib.as_closest_canonical(mri)
mri_im = mri.get_fdata()
# inverse the contrast of this weird MRI so that the background is black
mri_im = mri_im.max() - mri_im

# original array shape for all the images
common_shape = mri_im.shape

# MRI with artificially added skull
mri_skull = nib.load(os.path.join(sdir,'mri_registered_with_skull_2009b_sym.nii'))
mri_skull = nib.as_closest_canonical(mri_skull)
mri_im_skull = mri_skull.get_fdata()
# inverse the contrast of this weird MRI so that the background is black
mri_im_skull = mri_im_skull.max() - mri_im_skull

# Hammersmith brain atlas, probability maps, no csf
# pair contralateral regions to ensure the same image parameters
hammer_prob_files_L = sorted(glob(os.path.join(sdir, 'Hammersmith', '*_L*.nii')))
hammer_prob_files_R_only = sorted(glob(os.path.join(sdir, 'Hammersmith', '*_R*.nii')))
# normally no left only nor right only regions
hammer_prob_files_LR = []
hammer_prob_files_L_only = []
for i,f in enumerate(hammer_prob_files_L):
    temp = f.split('.nii.gz_2009b.nii')[0].split('-')[-1]
    temp = temp.replace('_L','_R')
    contralat = glob(os.path.join(sdir, 'Hammersmith', '*'+temp+'*.nii'))
    if contralat and len(contralat)==1:
        hammer_prob_files_LR.append([f, contralat[0]])
        hammer_prob_files_R_only.remove(contralat[0])
    else:
        hammer_prob_files_L_only.append(f)
if len(hammer_prob_files_L_only) > 1 or len(hammer_prob_files_R_only) > 1:
    raise NotImplementedError

# number of regions with possibly different parameters
nb_regions = len(hammer_prob_files_LR)

# add CSF to tissues classes
if build_csf:
    # whole brain mask including approximate CSF
    mask_whole = binary_fill_holes( binary_closing( binary_dilation( classes_im>1, iterations=3), iterations=6))
    mask_whole = binary_closing( binary_dilation( mask_whole, iterations=3), iterations=10)
    # extract CSF
    csf =  mask_whole.astype(np.float64) - (classes_im>1).astype(np.float64)
    csf[csf<0] = 0
    csf = csf.astype(bool)
    # add CSF to the tissue classes image
    classes_full_im = classes_im.copy()
    classes_full_im[csf==True] = 10
    # save the new tissue classes image
    cls_nii = nib.Nifti1Image(classes_full_im, classes.affine, classes.header)
    nib.save(cls_nii, os.path.join(sdir, 'full_cls_400um_2009b_sym_with_csf_mfilip.nii'))

# tissue classes with added CSF
classes_full = nib.load(os.path.join(sdir, 'full_cls_400um_2009b_sym_with_csf_mfilip.nii'))
classes_full = nib.as_closest_canonical(classes_full)
classes_full_im = classes_full.get_fdata().astype(np.uint8)

# extract different tissue classes
wm = (classes_full_im==3).astype(np.float64)
gm = (classes_full_im==2).astype(np.float64)
cerv = (classes_full_im==4).astype(np.float64)
gm_sup = (classes_full_im==5).astype(np.float64)
nuclei = (classes_full_im==6).astype(np.float64)
amygd = (classes_full_im==7).astype(np.float64)
cerv_sup = (classes_full_im==8).astype(np.float64)
stem = (classes_full_im==9).astype(np.float64)
csf = (classes_full_im==10).astype(np.float64)

# simplify regions as no idea what should be the Na characteristics for some regions
gm_full = gm + gm_sup + cerv + cerv_sup
no_idea = nuclei + amygd + stem

# add a lesion if required
if add_lesion:
    # load a region in GM and WM
    lesion = nib.load(os.path.join(sdir, 'Hammersmith/probmap-full-r12-TL_superior_temporal_gyrus_middle_part_L.nii.gz_2009b.nii'))
    lesion = nib.as_closest_canonical(lesion)
    lesion = lesion.get_fdata()
    
    # reduce the size of the region asymmetrically
    lesion = grey_erosion(lesion, size=(1,20,10))
    lesion /= lesion.max()
    lesion[lesion<0.7]=0.
    # limit to GM and WM
    lesion *= gm_full + wm

# draw randomly T2* and Na concentration values for each region
# from Gaussians truncated to realistic intervals
loc, scale = 3., 1.
a, b = (0.3 - loc) / scale, (6 - loc) / scale
t2bi_s = truncnorm.rvs(a, b, loc, scale, size = nb_regions)

loc, scale = 22., 2.
a, b = (15. - loc) / scale, (30. - loc) / scale
t2bi_l = truncnorm.rvs(a, b, loc, scale, size = nb_regions)

loc, scale = 25., 2.
a, b = (20. - loc) / scale, (40. - loc) / scale
t2mono_l = truncnorm.rvs(a, b, loc, scale, size = nb_regions + 1)

loc, scale = 0.6, 0.1
a, b = (0.2 - loc) / scale, (0.8 - loc) / scale
t2bi_frac_l = truncnorm.rvs(a, b, loc, scale, size = nb_regions)

loc, scale = 140., 3.
a, b = (130. - loc) / scale, (150. - loc) / scale
conc_mono = truncnorm.rvs(a, b, loc, scale, size = nb_regions + 1)

loc, scale = 15., 3.
a, b = (5. - loc) / scale, (30. - loc) / scale
conc_bi = truncnorm.rvs(a, b, loc, scale, size = nb_regions)

# volume fraction of fluid (CSF and blood, though the blood may be slightly biexponential), with monoexponential T2* relaxation
v_mono = csf + gm_full * 0.1 + wm * 0.2 + no_idea * 0.05
if add_lesion:
    v_mono += lesion * 0.4

# the rest is the volume fraction of Na with biexponential T2* relaxation
v_bi = 1 - v_mono

# volume fraction weighted Na concentration for the biexponential compartment
final_vconc_bi = np.zeros(common_shape, np.float64)
# volume fraction weighted Na concentration for the monoexponential compartment
final_vconc_mono = np.zeros(common_shape, np.float64)
# T2* spatial map of the short component in the biexponential compartment
final_t2_bi_s = np.zeros(common_shape, np.float64)
# T2* spatial map of the long component in the biexponential compartment
final_t2_bi_l = np.zeros(common_shape, np.float64)
# fraction of the long T2* component in the biexponential compartment
final_t2_bi_frac_l = np.zeros(common_shape, np.float64)
# T2* spatial map of the (one and only) long component in the monoexponential compartment
final_t2_mono_l = np.zeros(common_shape, np.float64)

# compute the final phantom
if compute_phantom:
    # load hammersmith region probability maps
    for i,f in enumerate(hammer_prob_files_LR):
        # read left region
        temp = nib.load(os.path.join(sdir, 'Hammersmith', f[0]))
        temp = nib.as_closest_canonical(temp)
        hammer_prob_im = temp.get_fdata()
        # read and add right region
        temp = nib.load(os.path.join(sdir, 'Hammersmith', f[1]))
        temp = nib.as_closest_canonical(temp)
        hammer_prob_im += temp.get_fdata()

        # divide by the (should be) global maximum
        if i==0:
            mm = hammer_prob_im.max()
        hammer_prob_im /= mm

        # compute and add parameters for each region
        final_vconc_bi += v_bi * conc_bi[i] * hammer_prob_im * ( wm * 0.9 + gm_full + no_idea * 0.1 )
        final_vconc_mono += v_mono * conc_mono[i] * hammer_prob_im * ( wm * 0.9 + gm_full + no_idea * 0.1)
        final_t2_bi_s += t2bi_s[i] * hammer_prob_im
        final_t2_bi_l += t2bi_l[i] * hammer_prob_im
        final_t2_bi_frac_l += t2bi_frac_l[i] * hammer_prob_im
        final_t2_mono_l += t2mono_l[i] * hammer_prob_im

    # add csf region
    final_vconc_mono += conc_mono[i] * csf
    final_t2_mono_l += t2mono_l[i] * csf
    final_vconc_total = final_vconc_bi + final_vconc_mono

    # final Na phantom parameters in initial high resolution
    np.save(os.path.join(odir, 'vconc_bi'), final_vconc_bi)
    np.save(os.path.join(odir, 'vconc_mono'), final_vconc_mono)
    np.save(os.path.join(odir, 't2bi_s'), final_t2_bi_s)
    np.save(os.path.join(odir, 't2bi_l'), final_t2_bi_l)
    np.save(os.path.join(odir, 't2mono_l'), final_t2_mono_l)
    np.save(os.path.join(odir, 't2bi_frac_l'), final_t2_bi_frac_l)
    np.save(os.path.join(odir, 'Hmri'), mri_im)

    # pad to cubic volume
    max_n = max(common_shape)
    pad_shape = list(common_shape)
    pad_shape = [ [(max_n - x)//2, (max_n - x)//2]  for x in pad_shape ]

    final_vconc_bi = np.pad(final_vconc_bi, pad_shape)
    final_vconc_mono = np.pad(final_vconc_mono, pad_shape)
    final_vconc_total = np.pad(final_vconc_total, pad_shape)
    final_t2_bi_s = np.pad(final_t2_bi_s, pad_shape)
    final_t2_bi_l = np.pad(final_t2_bi_l, pad_shape)
    final_t2_mono_l = np.pad(final_t2_mono_l, pad_shape)
    final_t2_bi_frac_l = np.pad(final_t2_bi_frac_l, pad_shape)
    final_mri = np.pad(mri_im, pad_shape)

    # downsample to the voxel size used for reconstructing real Na MRI TPI data
    ds = final_n/max_n
    final_vconc_bi_final_n = zoom(final_vconc_bi, ds, order=1)
    final_vconc_mono_final_n = zoom(final_vconc_mono, ds, order=1)
    final_vconc_total_final_n = zoom(final_vconc_total, ds, order=1)
    final_t2_bi_s_final_n = zoom(final_t2_bi_s, ds, order=1)
    final_t2_bi_l_final_n = zoom(final_t2_bi_l, ds, order=1)
    final_t2_mono_l_final_n = zoom(final_t2_mono_l, ds, order=1)
    final_t2_bi_frac_l_final_n = zoom(final_t2_bi_frac_l, ds, order=1)
    final_mri_final_n = zoom(final_mri, ds, order=1)

    # save downsampled parameters
    np.save(os.path.join(odir, f"vconc_bi_{final_n}"), final_vconc_bi_final_n)
    np.save(os.path.join(odir, f"vconc_mono_{final_n}"), final_vconc_mono_final_n)
    np.save(os.path.join(odir, f"t2bi_s_{final_n}"), final_t2_bi_s_final_n)
    np.save(os.path.join(odir, f"t2bi_l_{final_n}"), final_t2_bi_l_final_n)
    np.save(os.path.join(odir, f"t2mono_l_{final_n}"), final_t2_mono_l_final_n)
    np.save(os.path.join(odir, f"t2bi_frac_l_{final_n}"), final_t2_bi_frac_l_final_n)
    np.save(os.path.join(odir, f"Hmri_{final_n}"), final_mri_final_n)

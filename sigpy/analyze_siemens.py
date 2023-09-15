import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import find_objects, binary_dilation, binary_erosion
import SimpleITK as sitk
import pandas as pd

from collections import OrderedDict
from pathlib import Path
from utils import numpy_volume_to_sitk_image, sitk_image_to_numpy_volume


def resample_sodium_to_t1_grid(na_image,
                               na_voxsize,
                               na_origin,
                               t1,
                               t1_voxsize,
                               t1_origin,
                               final_transform,
                               missing=0.0):
    fixed_sitk_image = numpy_volume_to_sitk_image(na_image.astype(np.float32),
                                                  na_voxsize, na_origin)
    moving_sitk_image = numpy_volume_to_sitk_image(
        np.flip(t1, (0, 2)).astype(np.float32), t1_voxsize, t1_origin)

    x_sitk = sitk.Resample(fixed_sitk_image, moving_sitk_image,
                           final_transform.GetInverse(), sitk.sitkLinear,
                           missing, fixed_sitk_image.GetPixelID())

    return np.flip(sitk_image_to_numpy_volume(x_sitk), (0, 2))


#-----------------------------------------------------------------------------
pdict = OrderedDict({'31': [96, 134], '32': [78, 123], '37': [99, 131]})
sdir = 'recons_128'
agr_w_decay_file = 'agr_both_echo_w_decay_model_L1_3.0E-03_3.0E-01_2000_19.npz'
agr_wo_decay_file = 'agr_echo_1_no_decay_model_L1_3.0E-03_2000.npz'
agr_wo_decay_file_2 = 'agr_echo_2_no_decay_model_L1_3.0E-03_2000.npz'
ratio_file = 'est_ratio_3.0E-03_3.0E-01_100_19.npy'
conv_file = 'recon_echo_1_no_decay_model_L2_1.0E-01_2000.npz'
conv_file_2 = 'recon_echo_2_no_decay_model_L2_1.0E-01_2000.npz'

dfs = OrderedDict()

plt.style.use('dark_background')

for ip, (pnum, [slz, sly]) in enumerate(pdict.items()):

    pdir = Path(f'/data/sodium_mr/NYU/CSF{pnum}_raw')

    # load the t1
    t1_nii = nib.as_closest_canonical(nib.load(pdir / 't1.nii'))
    t1 = t1_nii.get_fdata()

    t1_affine = t1_nii.affine
    t1_voxsize = t1_nii.header['pixdim'][1:4]
    t1_origin = t1_affine[:-1, -1]

    # load the freesurfer seg
    aparc_nii = nib.as_closest_canonical(
        nib.load(pdir / 'aparc+aseg-native.nii'))
    aparc = aparc_nii.get_fdata()

    # load the transform that aligns the T1 to the sodium recons
    na_origin = np.loadtxt(pdir / sdir / 'na_origin.txt')
    final_transform = sitk.ReadTransform(str(pdir / sdir / 't1_transform.tfm'))

    # load the AGR including decay model
    na_voxsize = 10 * 22 / np.array([128, 128, 128])
    na_origin = t1_origin.copy()

    # load and resample the sodium recons
    conv = resample_sodium_to_t1_grid(
        np.abs(np.load(pdir / sdir / conv_file)['x']), na_voxsize, na_origin,
        t1, t1_voxsize, t1_origin, final_transform)
    
    conv_2 = resample_sodium_to_t1_grid(
        np.abs(np.load(pdir / sdir / conv_file_2)['x']), na_voxsize, na_origin,
        t1, t1_voxsize, t1_origin, final_transform)

    agr_wo_decay = resample_sodium_to_t1_grid(
        np.abs(np.load(pdir / sdir / agr_wo_decay_file)['x']), na_voxsize,
        na_origin, t1, t1_voxsize, t1_origin, final_transform)

    agr_wo_decay_2 = resample_sodium_to_t1_grid(
        np.abs(np.load(pdir / sdir / agr_wo_decay_file_2)['x']), na_voxsize,
        na_origin, t1, t1_voxsize, t1_origin, final_transform)

    agr_w_decay = resample_sodium_to_t1_grid(
        np.abs(np.load(pdir / sdir / agr_w_decay_file)['x']), na_voxsize,
        na_origin, t1, t1_voxsize, t1_origin, final_transform)

    est_ratio = resample_sodium_to_t1_grid(np.abs(
        np.load(pdir / sdir / ratio_file)),
                                           na_voxsize,
                                           na_origin,
                                           t1,
                                           t1_voxsize,
                                           t1_origin,
                                           final_transform,
                                           missing=1.0)

    #    #------------------------------------------------------------------------------
    #    # export to nifti
    #    edir = pdir / 'export'
    #    edir.mkdir(exist_ok=True)
    #
    #    nib.save(nib.Nifti1Image(t1.astype(np.float32), t1_nii.affine),
    #             edir / 'T1.nii')
    #    nib.save(nib.Nifti1Image(conv, t1_nii.affine), edir / 'CR.nii')
    #    nib.save(nib.Nifti1Image(agr_wo_decay, t1_nii.affine), edir / 'AGR.nii')
    #    nib.save(nib.Nifti1Image(agr_w_decay, t1_nii.affine), edir / 'AGRdm.nii')
    #    nib.save(nib.Nifti1Image(est_ratio, t1_nii.affine),
    #             edir / 'est_ratio_r.nii')

    #------------------------------------------------------------------------------
    # flip all images to LPS

    t1 = np.flip(t1, (0, 1))
    conv = np.flip(conv, (0, 1))
    conv_2 = np.flip(conv_2, (0, 1))
    agr_wo_decay = np.flip(agr_wo_decay, (0, 1))
    agr_wo_decay_2 = np.flip(agr_wo_decay_2, (0, 1))
    agr_w_decay = np.flip(agr_w_decay, (0, 1))
    est_ratio = np.flip(est_ratio, (0, 1))
    aparc = np.flip(aparc, (0, 1))

    # calculate the effective estimate T2star
    eff_T2star = np.zeros(est_ratio.shape)
    eff_T2star[est_ratio < 1] = -4.5 / np.log(est_ratio[est_ratio < 1])
    eff_T2star[t1 < 0.01*t1.max()] = 0


    #------------------------------------------------------------------------------
    # flip define ROIs and quantify

    agr_wo_means = OrderedDict()
    agr_w_means = OrderedDict()
    conv_means = OrderedDict()

    roi_inds = OrderedDict()
    cortical_gm_mask = (aparc >= 1000).astype(np.uint8)
    wm_mask = (aparc == 2).astype(np.uint8) + (aparc == 41).astype(np.uint8)
    # WM next to GM
    wm_gm_mask = binary_dilation(cortical_gm_mask, iterations=2) * wm_mask
    central_wm_mask = binary_erosion(wm_mask, iterations=5)
    ventricle_mask = (aparc == 4).astype(np.uint8) + (aparc == 43).astype(
        np.uint8)

    # inner brain stem - comparison to Haeger 2022
    brainstem_mask = binary_erosion((aparc == 16).astype(np.uint8),
                                    iterations=4)

    roi_inds['cortical GM'] = np.where(cortical_gm_mask)
    roi_inds['WM'] = np.where(wm_mask)
    roi_inds['brainstem'] = np.where(brainstem_mask)

    # normalize the images to the 135 mmol/L in the ventricles
    # for the conv recon of the 1st recon

    norm_fac = 135/conv[np.where(binary_erosion(ventricle_mask, iterations=2))].mean()

    conv *= norm_fac
    conv_2 *= norm_fac
    agr_wo_decay *= norm_fac
    agr_wo_decay_2 *= norm_fac
    agr_w_decay *= norm_fac

    for roi, inds in roi_inds.items():
        if ip == 0:
            dfs[roi] = pd.DataFrame()

        tmp = pd.DataFrame(
            {
                'CR': conv[inds].mean(),
                'AGR': agr_wo_decay[inds].mean(),
                'AGRdm': agr_w_decay[inds].mean()
            },
            index=[ip + 1])

        dfs[roi] = pd.concat((dfs[roi], tmp))

    #------------------------------------------------------------------------------

    bbox = find_objects(t1 > 0.1 * np.percentile(t1, 99.9))[0]

    #vmax = np.percentile(agr_w_decay, 99.99)
    vmax = 210.
    vmax_T2star = np.percentile(eff_T2star[aparc > 0], 99)
    tmax = np.percentile(t1, 99.9)

    # bounding box values for inset plot of transaxial slice
    xl = find_objects(aparc[bbox][:, :, slz].T > 0)[0][1].start
    xr = find_objects(aparc[bbox][:, :, slz].T > 0)[0][1].stop
    xm = (xl + xr) // 2
    ybottom = find_objects(aparc[bbox][:, :, slz].T > 0)[0][0].stop + 4

    # bounding box values for inset plot of coronal slice
    xl2 = find_objects(aparc[bbox][:, sly, :].T > 0)[0][1].start
    xr2 = find_objects(aparc[bbox][:, sly, :].T > 0)[0][1].stop
    xm2 = (xl + xr) // 2
    ytop2 = find_objects(aparc[bbox][:, sly, :].T > 0)[0][0].stop + 4

    fig, ax = plt.subplots(4, 5, figsize=(5 * 2, 4 * 2 + 1.5))

    i0 = ax[0, 0].imshow(t1[bbox][:, :, slz].T,
                         cmap='Greys_r',
                         vmin=0,
                         vmax=tmax)
    ax[1, 0].imshow(t1[bbox][:, sly, :].T,
                    origin='lower',
                    cmap='Greys_r',
                    vmin=0,
                    vmax=tmax)

    axins = ax[0, 0].inset_axes([0.85, 0.7, 0.35, 0.35])
    axins.imshow(t1[bbox][:, :, slz].T, cmap='Greys_r', vmin=0, vmax=tmax)
    axins.set_axis_off()
    axins.set_xlim(xm - 15, xm + 15)
    axins.set_ylim(ybottom, ybottom - 30)
    ax[0, 0].indicate_inset_zoom(axins, edgecolor="white")

    axins = ax[1, 0].inset_axes([0.85, 0.7, 0.35, 0.35])
    axins.imshow(t1[bbox][:, sly, :].T,
                 cmap='Greys_r',
                 vmin=0,
                 vmax=tmax,
                 origin='lower')
    axins.set_axis_off()
    axins.set_xlim(xm2 - 15, xm2 + 15)
    axins.set_ylim(ytop2 - 30, ytop2)
    ax[1, 0].indicate_inset_zoom(axins, edgecolor="white")

    # show conventional recons
    i1 = ax[0, 1].imshow(conv[bbox][:, :, slz].T,
                         cmap='Greys_r',
                         vmin=0,
                         vmax=vmax)
    ax[1, 1].imshow(conv[bbox][:, sly, :].T,
                    origin='lower',
                    cmap='Greys_r',
                    vmin=0,
                    vmax=vmax)

    i12 = ax[2, 1].imshow(conv_2[bbox][:, :, slz].T,
                         cmap='Greys_r',
                         vmin=0,
                         vmax=vmax)
    ax[3, 1].imshow(conv_2[bbox][:, sly, :].T,
                    origin='lower',
                    cmap='Greys_r',
                    vmin=0,
                    vmax=vmax)

    axins = ax[0, 1].inset_axes([0.9, 0.7, 0.35, 0.35])
    axins.imshow(conv[bbox][:, :, slz].T, cmap='Greys_r', vmin=0, vmax=vmax)
    axins.set_axis_off()
    axins.set_xlim(xm - 15, xm + 15)
    axins.set_ylim(ybottom, ybottom - 30)
    ax[0, 1].indicate_inset_zoom(axins, edgecolor="white")

    axins = ax[1, 1].inset_axes([0.85, 0.7, 0.35, 0.35])
    axins.imshow(conv[bbox][:, sly, :].T,
                 cmap='Greys_r',
                 vmin=0,
                 vmax=vmax,
                 origin='lower')
    axins.set_axis_off()
    axins.set_xlim(xm2 - 15, xm2 + 15)
    axins.set_ylim(ytop2 - 30, ytop2)
    ax[1, 1].indicate_inset_zoom(axins, edgecolor="white")


    axins = ax[2, 1].inset_axes([0.9, 0.7, 0.35, 0.35])
    axins.imshow(conv_2[bbox][:, :, slz].T, cmap='Greys_r', vmin=0, vmax=vmax)
    axins.set_axis_off()
    axins.set_xlim(xm - 15, xm + 15)
    axins.set_ylim(ybottom, ybottom - 30)
    ax[2, 1].indicate_inset_zoom(axins, edgecolor="white")

    axins = ax[3, 1].inset_axes([0.85, 0.7, 0.35, 0.35])
    axins.imshow(conv_2[bbox][:, sly, :].T,
                 cmap='Greys_r',
                 vmin=0,
                 vmax=vmax,
                 origin='lower')
    axins.set_axis_off()
    axins.set_xlim(xm2 - 15, xm2 + 15)
    axins.set_ylim(ytop2 - 30, ytop2)
    ax[3, 1].indicate_inset_zoom(axins, edgecolor="white")



    # show AGRs without decay model
    i2 = ax[0, 2].imshow(agr_wo_decay[bbox][:, :, slz].T,
                         cmap='Greys_r',
                         vmin=0,
                         vmax=vmax)
    ax[1, 2].imshow(agr_wo_decay[bbox][:, sly, :].T,
                    origin='lower',
                    cmap='Greys_r',
                    vmin=0,
                    vmax=vmax)

    i22 = ax[2, 2].imshow(agr_wo_decay_2[bbox][:, :, slz].T,
                         cmap='Greys_r',
                         vmin=0,
                         vmax=vmax)
    ax[3, 2].imshow(agr_wo_decay_2[bbox][:, sly, :].T,
                    origin='lower',
                    cmap='Greys_r',
                    vmin=0,
                    vmax=vmax)

    axins = ax[0, 2].inset_axes([0.85, 0.7, 0.35, 0.35])
    axins.imshow(agr_wo_decay[bbox][:, :, slz].T,
                 cmap='Greys_r',
                 vmin=0,
                 vmax=vmax)
    axins.set_axis_off()
    axins.set_xlim(xm - 15, xm + 15)
    axins.set_ylim(ybottom, ybottom - 30)
    ax[0, 2].indicate_inset_zoom(axins, edgecolor="white")

    axins = ax[1, 2].inset_axes([0.85, 0.7, 0.35, 0.35])
    axins.imshow(agr_wo_decay[bbox][:, sly, :].T,
                 cmap='Greys_r',
                 vmin=0,
                 vmax=vmax,
                 origin='lower')
    axins.set_axis_off()
    axins.set_xlim(xm2 - 15, xm2 + 15)
    axins.set_ylim(ytop2 - 30, ytop2)
    ax[1, 2].indicate_inset_zoom(axins, edgecolor="white")

    axins = ax[2, 2].inset_axes([0.85, 0.7, 0.35, 0.35])
    axins.imshow(agr_wo_decay_2[bbox][:, :, slz].T,
                 cmap='Greys_r',
                 vmin=0,
                 vmax=vmax)
    axins.set_axis_off()
    axins.set_xlim(xm - 15, xm + 15)
    axins.set_ylim(ybottom, ybottom - 30)
    ax[2, 2].indicate_inset_zoom(axins, edgecolor="white")

    axins = ax[3, 2].inset_axes([0.85, 0.7, 0.35, 0.35])
    axins.imshow(agr_wo_decay_2[bbox][:, sly, :].T,
                 cmap='Greys_r',
                 vmin=0,
                 vmax=vmax,
                 origin='lower')
    axins.set_axis_off()
    axins.set_xlim(xm2 - 15, xm2 + 15)
    axins.set_ylim(ytop2 - 30, ytop2)
    ax[3, 2].indicate_inset_zoom(axins, edgecolor="white")




    i3 = ax[0, 3].imshow(agr_w_decay[bbox][:, :, slz].T,
                         cmap='Greys_r',
                         vmin=0,
                         vmax=vmax)

    ax[1, 3].imshow(agr_w_decay[bbox][:, sly, :].T,
                    origin='lower',
                    cmap='Greys_r',
                    vmin=0,
                    vmax=vmax)

    axins = ax[0, 3].inset_axes([0.85, 0.7, 0.35, 0.35])
    axins.imshow(agr_w_decay[bbox][:, :, slz].T,
                 cmap='Greys_r',
                 vmin=0,
                 vmax=vmax)
    axins.set_axis_off()
    axins.set_xlim(xm - 15, xm + 15)
    axins.set_ylim(ybottom, ybottom - 30)
    ax[0, 3].indicate_inset_zoom(axins, edgecolor="white")

    axins = ax[1, 3].inset_axes([0.85, 0.7, 0.35, 0.35])
    axins.imshow(agr_w_decay[bbox][:, sly, :].T,
                 cmap='Greys_r',
                 vmin=0,
                 vmax=vmax,
                 origin='lower')
    axins.set_axis_off()
    axins.set_xlim(xm2 - 15, xm2 + 15)
    axins.set_ylim(ytop2 - 30, ytop2)
    ax[1, 3].indicate_inset_zoom(axins, edgecolor="white")

    # show estimated ratio and effective T2star

    i4 = ax[0, 4].imshow(est_ratio[bbox][:, :, slz].T,
                         cmap='Greys_r',
                         vmin=0,
                         vmax=1)
    ax[1, 4].imshow(est_ratio[bbox][:, sly, :].T,
                    origin='lower',
                    cmap='Greys_r',
                    vmin=0,
                    vmax=1)

    i42 = ax[2, 4].imshow(eff_T2star[bbox][:, :, slz].T,
                         cmap='magma',
                         vmin=0,
                         vmax=vmax_T2star)
    ax[3, 4].imshow(eff_T2star[bbox][:, sly, :].T,
                    origin='lower',
                    cmap='magma',
                    vmin=0,
                    vmax=vmax_T2star)



    cax0 = fig.add_axes([0.02, 0.52, 0.18, 0.01])
    cb0 = fig.colorbar(i0, cax=cax0, orientation='horizontal')
    cb0.ax.tick_params(labelsize='small')
    cax1 = fig.add_axes([0.02 + 1 * 0.196, 0.52, 0.18, 0.01])
    cb1 = fig.colorbar(i1, cax=cax1, orientation='horizontal')
    cb1.ax.tick_params(labelsize='small')
    cax2 = fig.add_axes([0.02 + 2 * 0.196, 0.52, 0.18, 0.01])
    cb2 = fig.colorbar(i2, cax=cax2, orientation='horizontal')
    cb2.ax.tick_params(labelsize='small')
    cax3 = fig.add_axes([0.02 + 3 * 0.196, 0.52, 0.18, 0.01])
    cb3 = fig.colorbar(i3, cax=cax3, orientation='horizontal')
    cb3.ax.tick_params(labelsize='small')
    cax4 = fig.add_axes([0.02 + 4 * 0.196, 0.52, 0.18, 0.01])
    cb4 = fig.colorbar(i4, cax=cax4, orientation='horizontal')
    cb4.ax.tick_params(labelsize='small')

    cax12 = fig.add_axes([0.02 + 1 * 0.196, 0.035, 0.18, 0.01])
    cb12 = fig.colorbar(i12, cax=cax12, orientation='horizontal')
    cb12.ax.tick_params(labelsize='small')
    cax22 = fig.add_axes([0.02 + 2 * 0.196, 0.035, 0.18, 0.01])
    cb22 = fig.colorbar(i22, cax=cax22, orientation='horizontal')
    cb22.ax.tick_params(labelsize='small')
    cax42 = fig.add_axes([0.02 + 4 * 0.196, 0.035, 0.18, 0.01])
    cb42 = fig.colorbar(i42, cax=cax42, orientation='horizontal')
    cb42.ax.tick_params(labelsize='small')

    for axx in ax.ravel():
        axx.set_axis_off()

    ax[0, 0].set_title('1H T1', fontsize='small')
    ax[0, 1].set_title('23Na CR, TE=0.5ms (mmol/L)', fontsize='small')
    ax[0, 2].set_title('23Na AGR, TE=0.5ms (mmol/L)', fontsize='small')
    ax[0, 3].set_title('23Na AGRdm (mmol/L)', fontsize='small')
    ax[0, 4].set_title('est. ratio r', fontsize='small')

    ax[2, 1].set_title('23Na CR, TE=5ms (mmol/L)', fontsize='small')
    ax[2, 2].set_title('23Na AGR, TE=5ms (mmol/L)', fontsize='small')
    ax[2, 4].set_title('est. eff. T2* (ms)', fontsize='small')

    fig.tight_layout(pad=1.8)
    fig.show()

#------------------------------------------------------------------------------

ratio_df = dfs['cortical GM'] / dfs['WM']
ratio_df_brainstem = dfs['cortical GM'] / dfs['brainstem']

plt.style.use('default')

fig2, ax2 = plt.subplots(1, 5, figsize=(12, 3))
dfs['cortical GM'].plot(ax=ax2[0], rot=0, kind='bar', legend=False)
dfs['WM'].plot(ax=ax2[1], rot=0, kind='bar', legend=False)
dfs['brainstem'].plot(ax=ax2[2], rot=0, kind='bar', legend=False)

ratio_df.plot(ax=ax2[3], rot=0, kind='bar', legend=False)
ratio_df_brainstem.plot(ax=ax2[4], rot=0, kind='bar').legend(loc='lower right')

ax2[0].set_title('GM', fontsize='medium')
ax2[1].set_title('WM', fontsize='medium')
ax2[2].set_title('brainstem center', fontsize='medium')

ax2[3].set_title('GM / WM', fontsize='medium')
ax2[4].set_title('GM / brainstem center', fontsize='medium')

for i in range(3):
    ax2[i].set_ylabel('sodium concentration (mmol/L)', fontsize='medium')
    ax2[i].set_ylim(0, 0.95*norm_fac)

ax2[3].set_ylabel('sodium concentration ratio')
ax2[4].set_ylabel('sodium concentration ratio')

for axx in ax2.ravel():
    axx.set_xlabel('subject')
    axx.grid(ls=':')

fig2.tight_layout()
fig2.show()
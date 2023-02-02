from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import pymirc.viewer as pv

noise_level = 2e5
config_files = sorted(list(Path('run_10_iter').rglob('config.json')))

df = pd.DataFrame()

for i, config_file in enumerate(config_files):
    #print(config_file)

    with open(config_file, 'r') as f:
        config_data = json.load(f)

    tmp = pd.DataFrame(config_data, index=[i])
    run_dir = config_file.parent
    tmp["run_dir"] = run_dir
    df = pd.concat([df, tmp])

    # load data
    if i > 1000:
        ifft_echo_1_corr = np.load(run_dir / 'ifft_echo_1_corr.npy')
        ifft_echo_2_corr = np.load(run_dir / 'ifft_echo_2_corr.npy')
        ifft_echo_1_filtered_corr = np.load(run_dir /
                                            'ifft_echo_1_filtered_corr.npy')
        ifft_echo_2_filtered_corr = np.load(run_dir /
                                            'ifft_echo_2_filtered_corr.npy')
        agr_na = np.load(run_dir / 'agr_na.npy')
        Gam_recon = np.load(run_dir / 'gamma.npy')
        aimg = np.load(run_dir / 'anatomical_prior_image.npy')
        na_img = np.load(run_dir / 'true_na_image.npy')
        t1_image = np.load(run_dir / 't1_image.npy')
        corr_field = np.load(run_dir / 'corr_field.npy')

        vi = pv.ThreeAxisViewer(
            [np.abs(ifft_echo_1_filtered_corr),
             np.abs(agr_na), Gam_recon])

        dummy = input('')

df.sort_values(
    ['noise_level', 'beta_gamma', 'gradient_strength', 'beta_recon'],
    inplace=True)
df.reset_index(inplace=True, drop=True)

# select one data from one noise level
df = df.loc[df.noise_level == noise_level]

# convert colums to categorical
cats = ['noise_level', 'gradient_strength', 'beta_gamma', 'beta_recon']
df[cats] = df[cats].astype('category')

#------------------------------------------------------------------------------
# visualizations

ims_na = dict(origin='lower', vmax=2.5, cmap=plt.cm.Greys_r)
ims_gam = dict(origin='lower', vmin=0.5, vmax=1, cmap=plt.cm.Greys_r)
sl = 64

na_fig1, na_ax1 = plt.subplots(4, 4, figsize=(9, 9))
na_fig2, na_ax2 = plt.subplots(4, 4, figsize=(9, 9))
na_fig3, na_ax3 = plt.subplots(4, 4, figsize=(9, 9))

na_figs = [na_fig1, na_fig2, na_fig3]
na_axs = [na_ax1, na_ax2, na_ax3]

gam_fig1, gam_ax1 = plt.subplots(4, 4, figsize=(9, 9))
gam_fig2, gam_ax2 = plt.subplots(4, 4, figsize=(9, 9))
gam_fig3, gam_ax3 = plt.subplots(4, 4, figsize=(9, 9))

gam_figs = [gam_fig1, gam_fig2, gam_fig3]
gam_axs = [gam_ax1, gam_ax2, gam_ax3]

for (gradient_strength,
     beta_gamma), ddf in df.groupby(['gradient_strength', 'beta_gamma']):
    print(gradient_strength, beta_gamma)

    fig_index = df.gradient_strength.cat.categories.to_list().index(
        gradient_strength)
    row_index = df.beta_gamma.cat.categories.to_list().index(beta_gamma)

    for j in range(ddf.shape[0]):
        print(fig_index, row_index, j)
        run_dir = ddf.iloc[j].run_dir
        agr_na = np.load(run_dir / 'agr_na.npy')
        gam_recon = np.load(run_dir / 'gamma.npy')

        na_axs[fig_index][row_index,
                          j].imshow(np.abs(agr_na)[..., sl].T, **ims_na)
        gam_axs[fig_index][row_index, j].imshow(gam_recon[..., sl].T,
                                                **ims_gam)

        if row_index == 0:
            na_axs[fig_index][row_index, j].set_title(
                f'beta recon {ddf.iloc[j].beta_recon}')
            gam_axs[fig_index][row_index, j].set_title(
                f'beta recon {ddf.iloc[j].beta_recon}')

        if j == 0:
            na_axs[fig_index][row_index, j].set_ylabel(
                f'beta gamma {ddf.iloc[j].beta_gamma}')
            gam_axs[fig_index][row_index, j].set_ylabel(
                f'beta gamma {ddf.iloc[j].beta_gamma}')

for ax in na_axs:
    for axx in ax.ravel():
        axx.tick_params(left=False,
                        right=False,
                        labelleft=False,
                        labelbottom=False,
                        bottom=False)

for ax in gam_axs:
    for axx in ax.ravel():
        axx.tick_params(left=False,
                        right=False,
                        labelleft=False,
                        labelbottom=False,
                        bottom=False)

for i, fig in enumerate(na_figs):
    fig.suptitle(f'gradient strength {df.gradient_strength.cat.categories[i]}')
    fig.tight_layout()
    fig.show()

for i, fig in enumerate(gam_figs):
    fig.suptitle(f'gradient strength {df.gradient_strength.cat.categories[i]}')
    fig.tight_layout()
    fig.show()
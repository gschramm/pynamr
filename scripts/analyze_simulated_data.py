from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.ndimage import binary_dilation

import pymirc.viewer as pv
from pymirc.image_operations import zoom3d

import seaborn as sns

noise_level = 2e5
sim_path: Path = Path('run_10_iter')
config_files = sorted(list(sim_path.rglob('config.json')))

df = pd.DataFrame()

for i, config_file in enumerate(config_files):
    #print(config_file)

    with open(config_file, 'r') as f:
        config_data = json.load(f)

    tmp = pd.DataFrame(config_data, index=[i])
    run_dir = config_file.parent
    tmp["run_dir"] = run_dir
    df = pd.concat([df, tmp])

df.sort_values(
    ['noise_level', 'beta_gamma', 'gradient_strength', 'beta_recon'],
    inplace=True)
df.reset_index(inplace=True, drop=True)

# select one data from one noise level
df = df.loc[df.noise_level == noise_level]

# create an empty column for the GM/WM to
df['GM/WM ratio'] = 0.

# convert colums to categorical
cats = ['noise_level', 'gradient_strength', 'beta_gamma', 'beta_recon']
df[cats] = df[cats].astype('category')

# load the 256 GM, WM and cortex mask
gm_256 = np.load(sim_path / 'gm_256.npy')
wm_256 = np.load(sim_path / 'wm_256.npy')
cortex_256 = np.load(sim_path / 'cortex_256.npy')

# exclude subcortical region from GM mask
gm_256[cortex_256 == 0] = 0

gm_256_dilated = binary_dilation(gm_256, iterations=5).astype(int)
local_wm_256 = np.logical_and((gm_256_dilated - gm_256).astype(int),
                              wm_256.astype(int))

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
        agr_na = np.abs(np.load(run_dir / 'agr_na.npy'))
        gam_recon = np.load(run_dir / 'gamma.npy')

        # for the quantification we have to interpolate the reconstructed
        # image to the 256 grid
        agr_na_interp = zoom3d(agr_na, 2)
        gm_wm_ratio = agr_na_interp[gm_256].mean(
        ) / agr_na_interp[local_wm_256].mean()

        df['GM/WM ratio'][ddf.index[j]] = gm_wm_ratio

        na_axs[fig_index][row_index, j].imshow(agr_na[..., sl].T, **ims_na)
        gam_axs[fig_index][row_index, j].imshow(gam_recon[..., sl].T,
                                                **ims_gam)

        if row_index == 0:
            na_axs[fig_index][row_index, j].set_title(
                f'beta recon {ddf.iloc[j].beta_recon}, GM/WM {gm_wm_ratio:.2f}',
                fontsize='medium')
            gam_axs[fig_index][row_index, j].set_title(
                f'beta recon {ddf.iloc[j].beta_recon}', fontsize='medium')
        else:
            na_axs[fig_index][row_index,
                              j].set_title(f'GM/WM {gm_wm_ratio:.2f}',
                                           fontsize='medium')

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

# searborn grid plot for quantifiction
sns.set_context('paper')
sns.set_style('ticks')
grid = sns.FacetGrid(df, col='beta_gamma', hue='gradient_strength')
grid.map(sns.stripplot, 'beta_recon', 'GM/WM ratio')
grid.add_legend()

for ax in grid.axes.ravel():
    ax.grid(ls=':')
    ax.set_ylim(0.95, 1.55)
    ax.axhline(1.5, color='k', lw=0.5, ls='--')

grid.fig.show()

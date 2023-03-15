from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import time
from scipy.ndimage import binary_dilation
from numba import jit

from pymirc.image_operations import zoom3d

import seaborn as sns

import sys
import os

import pymirc.viewer as pv


start_time = time.time()

# parameters
folder = 'abstract'
jitter_suffix = ''
#folder = 'nodecay'
#folder = 'jitter'
#jitter_suffix = '_jit1'

load_nodecay = False
smallROI = False

# base dir
workdir = '/uz/data/Admin/ngeworkingresearch/MarinaFilipovic/BrainWeb_DiffGrad_Sim/'
analysis_results_dir = Path(workdir) / 'analysis_results'

recon_path: Path = Path(workdir) / f'run_{folder}'
phantom_path: Path = Path(workdir) / 'brainweb54'

config_files = sorted(list(recon_path.rglob('config.json')))

df = pd.DataFrame()
for i, config_file in enumerate(config_files):
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    tmp = pd.DataFrame(config_data, index=[i])
    # run dir
    run_dir = config_file.parent
    tmp["run_dir"] = run_dir
    # represent data for gradient 16 without decay as another gradient category (16nd)
    # for easier comparison
    tmp["gradient_strength"] = tmp["gradient_strength"].astype(str)
    if tmp["no_decay"].values[0]:
        if load_nodecay:
            assert(tmp["gradient_strength"].values[0]=="16")
            tmp["gradient_strength"]="16nd"
        else:
            continue
    df = pd.concat([df, tmp])

# sort
df.sort_values(
    ['noise_level', 'seed', 'beta_gamma', 'gradient_strength', 'beta_recon'],
    inplace=True)
df.reset_index(inplace=True, drop=True)

# convert colums to categorical
cats = ['noise_level', 'gradient_strength', 'beta_gamma', 'beta_recon', 'seed']
df[cats] = df[cats].astype('category')

# select a noise level and an example noise realization
noise_level = 2e5
realiz = 892509 #df.seed[0] #859456 #df.seed[0]

# select data from one noise level, currently only one anyway
df = df.loc[df.noise_level == noise_level]
# TODO to be tested
df['noise_level'] = df['noise_level'].cat.remove_unused_categories()

# select a subset of seeds
#entire_seeds = [realiz, 6545715]
#df = df[df.seed.isin([realiz])]
#df = df[df.seed.isin(entire_seeds)]
#df['seed'] = df['seed'].cat.remove_unused_categories()

# select a subset of beta values
#beta_recon_chosen = [0.3, 1.]
#beta_gamma_chosen = [10., 30.]
#df = df[df.beta_recon.isin(beta_recon_chosen)]
#df = df[df.beta_gamma.isin(beta_gamma_chosen)]
#df['beta_recon'] = df['beta_recon'].cat.remove_unused_categories()
#df['beta_gamma'] = df['beta_gamma'].cat.remove_unused_categories()

# load true simulated na image
true_na_image = np.load(Path(workdir) / f'brainweb_n28p4dt10g16_23Na_v1{jitter_suffix}' / 'simulated_nufft_data.npz')['na_image']
t1_image = np.load(Path(workdir) / f'brainweb_n28p4dt10g16_23Na_v1{jitter_suffix}' / 'simulated_nufft_data.npz')['t1_image']

# main parameters for reconstructions
params = ['gradient_strength', 'beta_gamma', 'beta_recon']

# criteria for evaluation
criteria = ['gm_wm_ratio', 'gm', 'wm', 'csf', 'wm_local']
# add columns for criteria
for c in criteria:
    df[c] = 0.

# currently database tailored for agr recons
# prepare for later addition of conventional recons
df['recon_type'] = 'agr'
df['nofilt'] = True

# number of categories for each param
nb_grad = len(df.gradient_strength.cat.categories)
nb_beta_gamma = len(df.beta_gamma.cat.categories)
nb_beta_recon = len(df.beta_recon.cat.categories)
nb_realiz = len(df.seed.cat.categories)
print('gradients')
print(df.gradient_strength.cat.categories.tolist())
print('beta_gamma')
print(df.beta_gamma.cat.categories.tolist())
print('beta_recon')
print(df.beta_recon.cat.categories.tolist())
print('seeds')
print(df.seed.cat.categories.tolist())

# there should be a single entry for each possible combination of categories
assert(df.shape[0] == nb_grad*nb_beta_gamma*nb_beta_recon*nb_realiz)

# load the 256 GM, WM and cortex mask
gm_256 = np.load(phantom_path / 'gm_256.npy')
wm_256 = np.load(phantom_path / 'wm_256.npy')
cortex_256 = np.load(phantom_path / 'cortex_256.npy')
csf_256 = np.load(phantom_path / 'csf_256.npy')

# exclude subcortical region from GM mask
gm_256[cortex_256 == 0] = 0
gm_256_dilated = binary_dilation(gm_256, iterations=5).astype(int)
wm_local_256 = np.logical_and((gm_256_dilated - gm_256).astype(int),
                              wm_256.astype(int))

# reduce huge ROIs to small ROIs
if smallROI:
    reduction_mask = np.zeros(gm_256.shape, np.bool_)
    reduction_mask[150:,150:,128] = True
    reduction_mask[160:,160:,129] = True
    reduction_mask[160:,160:,127] = True
    gm_256 *= reduction_mask
    wm_256 *= reduction_mask
    wm_local_256 *= reduction_mask
    csf_256 *= reduction_mask

# truth
true = {}
true['gm_wm_ratio'] = true_na_image[gm_256].mean() / true_na_image[wm_local_256].mean()
true['gm'] = true_na_image[gm_256].mean()
true['wm'] = true_na_image[wm_256].mean()
true['csf'] = true_na_image[csf_256].mean()
true['wm_local'] = true_na_image[wm_local_256].mean()


# images for an example realization
agr_na_realiz = np.zeros((nb_grad, nb_beta_gamma, nb_beta_recon, 256, 256, 256), np.float64)
gamma_na_realiz = np.zeros((nb_grad, nb_beta_gamma, nb_beta_recon, 256, 256, 256), np.float64)
conv_realiz = np.zeros((nb_grad, 256, 256, 256), np.float64)
agr_na_realiz_mean = np.zeros((nb_grad, nb_beta_gamma, nb_beta_recon, 256, 256, 256), np.float64)
#agr_na_realiz_std = np.zeros((nb_grad, nb_beta_gamma, nb_beta_recon, 256, 256, 256), np.float64)


# init database for conventional recons
df_conv = pd.DataFrame()

# load images, compute criteria and store results into the database
# too many images for storing all of them into an array
for r in range(df.shape[0]):
    run_dir = df.iloc[r].run_dir
    agr_na = np.abs(np.load(run_dir / 'agr_na.npy'))
    gam_recon = np.load(run_dir / 'gamma.npy')
    conv_nofilt = np.abs(np.load(run_dir / 'ifft_echo_1_corr.npy'))
    conv = np.abs(np.load(run_dir / 'ifft_echo_1_filtered_corr.npy'))

    # AGR
    # for the quantification we have to interpolate the reconstructed
    # image to the 256 grid
    agr_na_interp = zoom3d(agr_na, 2)
    gamma_na_interp = zoom3d(gam_recon, 2)
    # GM/WM ratio
    gm_wm_ratio = agr_na_interp[gm_256].mean(
        ) / agr_na_interp[wm_local_256].mean()
    df.at[df.index[r], 'gm_wm_ratio'] = float(gm_wm_ratio)
    # GM mean
    gm = agr_na_interp[gm_256].mean()
    df.at[df.index[r], 'gm'] = float(gm)
    # WM mean
    wm = agr_na_interp[wm_256].mean()
    df.at[df.index[r], 'wm'] = float(wm)
    # WM local mean
    wm_local = agr_na_interp[wm_local_256].mean()
    df.at[df.index[r], 'wm_local'] = float(wm_local)
    # CSF mean
    csf = agr_na_interp[csf_256].mean()
    df.at[df.index[r], 'csf'] = float(csf)

    # filtered conv recon
    # build db line
    df_conv_line = df.loc[df.index[r]].copy()
    df_conv_line['nofilt'] = False
    df_conv_line['recon_type'] = 'conv'

    conv_interp = zoom3d(conv, 2)
    gm_wm_ratio = conv_interp[gm_256].mean(
        ) / conv_interp[wm_local_256].mean()
    gm = conv_interp[gm_256].mean()
    wm = conv_interp[wm_256].mean()
    wm_local = conv_interp[wm_local_256].mean()
    csf = conv_interp[csf_256].mean()

    df_conv_line['gm_wm_ratio'] = float(gm_wm_ratio)
    df_conv_line['gm'] = float(gm)
    df_conv_line['wm'] = float(wm)
    df_conv_line['wm_local'] = float(wm_local)
    df_conv_line['csf'] = float(csf)

    # not filtered conv recon
    df_conv_line_filt = df_conv_line.copy()
    df_conv_line_filt['nofilt'] = True

    conv_interp = zoom3d(conv_nofilt, 2)
    gm_wm_ratio = conv_interp[gm_256].mean(
        ) / conv_interp[wm_local_256].mean()
    gm = conv_interp[gm_256].mean()
    wm = conv_interp[wm_256].mean()
    wm_local = conv_interp[wm_local_256].mean()
    csf = conv_interp[csf_256].mean()

    df_conv_line_filt['gm_wm_ratio'] = float(gm_wm_ratio)
    df_conv_line_filt['gm'] = float(gm)
    df_conv_line_filt['wm'] = float(wm)
    df_conv_line_filt['wm_local'] = float(wm_local)
    df_conv_line_filt['csf'] = float(csf)

    # add the filtered and not filtered conventional recon to conv db 
    df_conv = pd.concat([df_conv, df_conv_line.to_frame().T, df_conv_line_filt.to_frame().T], ignore_index=True)

    # save the images for the example realization for display
    if df.at[df.index[r], 'seed'] == realiz:
        grad_index = df.gradient_strength.cat.categories.to_list().index(
                df.at[df.index[r], 'gradient_strength'])
        beta_gamma_index = df.beta_gamma.cat.categories.to_list().index(df.at[df.index[r], 'beta_gamma'])
        beta_recon_index = df.beta_recon.cat.categories.to_list().index(df.at[df.index[r], 'beta_recon'])
        realiz_index = df.seed.cat.categories.to_list().index(df.at[df.index[r], 'seed'])
        agr_na_realiz[grad_index, beta_gamma_index, beta_recon_index] = agr_na_interp
        gamma_na_realiz[grad_index, beta_gamma_index, beta_recon_index] = gamma_na_interp
        conv_realiz[grad_index] = conv_interp

    # compute the mean over realizations
    agr_na_realiz_mean += agr_na_interp

# add the database with conventional reconstructions to the agr db
df = pd.concat([df, df_conv], ignore_index=True)
# apparently the categorical columns lost their categorical property after concatenation
cats = ['noise_level', 'gradient_strength', 'beta_gamma', 'beta_recon', 'seed', 'recon_type', 'nofilt']
df[cats] = df[cats].astype('category')

#------------------------------------------------------------------------------
# visualize single realization for single noise level

#ims_na = dict(origin='lower', vmax=2.5, cmap=plt.cm.Greys_r)
#ims_gam = dict(origin='lower', vmin=0.5, vmax=1, cmap=plt.cm.Greys_r)
ims_na = dict(origin='lower', vmax=3., cmap=plt.cm.viridis)
ims_gam = dict(origin='lower', vmin=0.5, vmax=1, cmap=plt.cm.viridis)
sl = 128 

# use data for a single realization
df_viz = df.loc[df.seed == realiz]

# show the truth
truth_fig, truth_ax = plt.subplots(figsize=(3,3))
truth_ax.imshow(true_na_image[:,:,true_na_image.shape[2]//2].T, **ims_na)
truth_fig.suptitle('Truth', fontsize='medium')
truth_ax.set_title(
                f"GM/WM {true['gm_wm_ratio']:.2f}",
                fontsize='small')
truth_fig.tight_layout()
truth_ax.axis('off')
truth_fig.show()

# show all the images
na_figs = []
na_axs = []
gam_figs = []
gam_axs = []
conv_figs = []
conv_axs = []
for g in range(nb_grad):
    temp_fig, temp_ax = plt.subplots(nb_beta_gamma, nb_beta_recon, figsize=(9, 9))
    na_figs.append(temp_fig)
    na_axs.append(temp_ax)

    temp_fig, temp_ax = plt.subplots(nb_beta_gamma, nb_beta_recon, figsize=(9, 9))
    gam_figs.append(temp_fig)
    gam_axs.append(temp_ax)

    temp_fig, temp_ax = plt.subplots(nb_beta_gamma, nb_beta_recon, figsize=(9, 9))
    conv_figs.append(temp_fig)
    conv_axs.append(temp_ax)

for fig_index in range(nb_grad):
    for row_index in range(nb_beta_recon):
        for r in range(nb_beta_gamma):

            na_axs[fig_index][row_index, r].imshow(agr_na_realiz[fig_index, r, row_index, ..., sl].T, **ims_na)
            gam_axs[fig_index][row_index, r].imshow(gamma_na_realiz[fig_index, r, row_index, ..., sl].T,
                                                    **ims_gam)
            conv_axs[fig_index][row_index, r].imshow(conv_realiz[fig_index, ..., sl].T, **ims_na)

            if row_index == 0:
                na_axs[fig_index][row_index, r].set_title(
                    f'beta recon {df.beta_recon.cat.categories.to_list()[r]}',
                    fontsize='small')
                gam_axs[fig_index][row_index, r].set_title(
                    f'beta recon {df.beta_recon.cat.categories.to_list()[r]}', fontsize='small')
                conv_axs[fig_index][row_index, r].set_title(
                    f'beta recon {df.beta_recon.cat.categories.to_list()[r]}',
                    fontsize='small')

            if r == 0:
                na_axs[fig_index][row_index, r].set_ylabel(
                    f'beta gamma {df.beta_gamma.cat.categories.to_list()[row_index]}', fontsize='small')
                gam_axs[fig_index][row_index, r].set_ylabel(
                    f'beta gamma {df.beta_gamma.cat.categories.to_list()[row_index]}', fontsize='small')
                conv_axs[fig_index][row_index, r].set_ylabel(
                    f'beta gamma {df.beta_gamma.cat.categories.to_list()[row_index]}', fontsize='small')

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

for ax in conv_axs:
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
    fig.savefig(Path(analysis_results_dir) / f'agr_grad{df.gradient_strength.cat.categories[i]}.png')

for i, fig in enumerate(gam_figs):
    fig.suptitle(f'gradient strength {df.gradient_strength.cat.categories[i]}')
    fig.tight_layout()
    fig.show()
    fig.savefig(Path(analysis_results_dir) / f'gamma_grad{df.gradient_strength.cat.categories[i]}.png')

for i, fig in enumerate(conv_figs):
    fig.suptitle(f'gradient strength {df.gradient_strength.cat.categories[i]}')
    fig.tight_layout()
    fig.show()
    fig.savefig(Path(analysis_results_dir) / f'conv_grad{df.gradient_strength.cat.categories[i]}.png')

# ----------------------------------------------------------------
# init seaborn 
sns.set_context('notebook')
sns.set(font_scale=1.5)
sns.set_style('ticks')

# ----------------------------------------------------------------
# some functions for analysis

# ----------------------------------------------------------------
# plot mean/stddev over realizations for each quantification criterion
# does not work anymore because database changed TODO
def plot_crit_mean_stddev(df: pd.DataFrame):
    # show criteria statistics one by one
    for col in criteria:
        current_data = df[[*params, col]]
        if '16nd' in current_data['gradient_strength'].cat.categories:
            current_data['gradient_strength'] = current_data['gradient_strength'].cat.reorder_categories(['16','24','32','16nd'])
        col_wrap = 4
        grid = sns.catplot(data=current_data, x="beta_recon", y=col, hue="gradient_strength", col='beta_gamma', kind="point", col_wrap=col_wrap, legend_out=False)
        grid.add_legend()

        # add limits, truth, criteria for simple (conventional) recon
        # conventional recon/criteria depend only on the gradient strength
        col_conv = col+'_conv'
        conv_data = df[['gradient_strength', col_conv]]
        min_val = min(min(current_data[col].min(), true[col]), conv_data[col_conv].min())
        max_val = max(max(current_data[col].max(), true[col]), conv_data[col_conv].max())
        min_val = min_val - 0.05 * np.abs(min_val)
        max_val = max_val + 0.05 * np.abs(max_val)
        legend_once = True
        for ax in grid.axes.ravel():
            ax.grid(ls=':')
            ax.set_ylim(min_val, max_val)
            ax.axhline(true[col], color='k', lw=2, ls='--', label='true')
            # plot conventional recon criteria for all gradient strengths
            # TODO plot them with the same color as for AGR recon, also a bit ugly
            for grad in conv_data.gradient_strength.cat.categories.to_list():
                conv_crit_value = conv_data[conv_data['gradient_strength']==grad][col_conv].mean() #.values[0]
                ls = ('-.' if grad=="16nd" else '--')
                label_conv = 'conv' if grad=='16' else ''
                label = ('conv nd' if grad=='16nd' else label_conv)
                ax.axhline(conv_crit_value, color='tab:gray', lw=2, ls=ls, label=label)
            if legend_once:
               ax.legend()
               legend_once = False

        grid.fig.show()
        grid.fig.savefig(Path(analysis_results_dir) / f'{folder}_{col}{jitter_suffix}_{nb_realiz}seeds.pdf')

# show  criteria difference wrt the truth one by one
# does not work anymore because database changed TODO
def plot_crit_mean_stddev_perc(df: pd.DataFrame):

    for col in criteria:
        current_data = df[[*params, col]].copy()
        if '16nd' in current_data['gradient_strength'].cat.categories:
            current_data['gradient_strength'] = current_data['gradient_strength'].cat.reorder_categories(['16','24','32','16nd'])
        current_data[col] = 100 * (current_data[col].values - true[col]) / true[col]
        current_true = 0.
        col_wrap = 4
        grid = sns.catplot(data=current_data, x="beta_recon", y=col, hue="gradient_strength", col='beta_gamma', kind="point", col_wrap=col_wrap, legend_out=False)
        grid.add_legend()

        # add limits, truth, criteria for simple (conventional) recon
        # conventional recon/criteria depend only on the gradient strength
        col_conv = col+'_conv'
        conv_data = df[['gradient_strength', col_conv]].copy()
        conv_data[col_conv] = 100 * (conv_data[col_conv].values - true[col]) / true[col]
        min_val = min(min(current_data[col].min(), current_true), conv_data[col_conv].min())
        max_val = max(max(current_data[col].max(), current_true), conv_data[col_conv].max())
        min_val = min_val - 5 
        max_val = max_val + 5
        legend_once = True
        for ax in grid.axes.ravel():
            ax.grid(ls=':')
            ax.set_ylim(min_val, max_val)
            ax.axhline(current_true, color='k', lw=2, ls='--', label='true')
            # plot conventional recon criteria for all gradient strengths
            # TODO plot them with the same color as for AGR recon, also a bit ugly
            for grad in conv_data.gradient_strength.cat.categories.to_list():
                conv_crit_value = conv_data[conv_data['gradient_strength']==grad][col_conv].mean() #.values[0]
                ls = ('-.' if grad=="16nd" else '--')
                label_conv = 'conv' if grad=='16' else ''
                label = ('conv nd' if grad=='16nd' else label_conv)
                ax.axhline(conv_crit_value, color='tab:gray', lw=2, ls=ls, label=label)
            if legend_once:
               ax.legend()
               legend_once = False

        grid.fig.show()
        grid.fig.savefig(Path(analysis_results_dir) / f'{folder}_{col}{jitter_suffix}_{nb_realiz}seeds_perc.pdf')

# = percentage bias/std 
def plot_crit_bias_std_perc(df: pd.DataFrame):
    # compute statistics (mean,std) for each parameter value combination, over realizations
    df_stats = df[params + criteria + [c+'_conv' for c in criteria] + [c+'_conv_nofilt' for c in criteria]].copy()
#    for i,c in enumerate(criteria):
#        df_stats[c] =  100 * (df_stats[c].values - true[c]) / true[c]
#        df_stats[c+'_conv'] =  100 * (df_stats[c+'_conv'].values - true[c]) / true[c]
#        df_stats[c+'_conv_filt'] =  100 * (df_stats[c+'_conv_filt'].values - true[c]) / true[c]
    # the number of rows for each group should be = number of seeds/realizations 
    ddf = df_stats.groupby(params)
    assert(np.all(ddf.count() == nb_realiz))
    # compute the mean and stddev
    df_stats = ddf.agg(['mean','std'])
    # flatten the index multiindex for use with FacetGrid
    df_stats = df_stats.reset_index()
    # flatten the columns multiindex for use with FacetGrid
    df_stats.columns = [' '.join(col).strip() for col in df_stats.columns.values]

    for i,c in enumerate(criteria):
        df_stats[c+' mean'] =  100 * (df_stats[c+' mean'].values - true[c]) / true[c]
        df_stats[c+'_conv mean'] =  100 * (df_stats[c+'_conv mean'].values - true[c]) / true[c]
        df_stats[c+'_conv_nofilt mean'] =  100 * (df_stats[c+'_conv_nofilt mean'].values - true[c]) / true[c]
        df_stats[c+' std'] =  100 * df_stats[c+' std'].values / true[c]
        df_stats[c+'_conv std'] =  100 * df_stats[c+'_conv std'].values / true[c]
        df_stats[c+'_conv_nofilt std'] =  100 * df_stats[c+'_conv_nofilt std'].values / true[c]
        df_stats = df_stats.rename(columns={c+' mean':c+' bias[%]', c+'_conv mean':c+'_conv bias[%]', c+'_conv_nofilt mean':c+'_conv_nofilt bias[%]'})
        df_stats = df_stats.rename(columns={c+' std':c+' std[%]', c+'_conv std':c+'_conv std[%]', c+'_conv_nofilt std':c+'_conv_nofilt std[%]'})


    for col in criteria:
        grid = sns.relplot(
            data=df_stats, kind="line",
            x=col+' bias[%]', y=col+' std[%]', hue="gradient_strength",
            col="beta_gamma", marker='.', markersize=10, legend='brief')
        # conv
        legend_once = True
        for i,ax in enumerate(grid.axes.ravel()):
            #ax.grid(ls=':')
            ax.set_ylim(0., max(df_stats[col+' std[%]'].max(), df_stats[col+'_conv_nofilt std[%]'].max())*1.2)
            ax.set_xlim(min(df_stats[col+' bias[%]'].min(), df_stats[col+'_conv bias[%]'].min())-1, max(df_stats[col+' bias[%]'].max(), df_stats[col+'_conv bias[%]'].max())+1)
            for grad in df_stats.gradient_strength.cat.categories.to_list():
                df_stats_conv_curr = df_stats[df_stats['gradient_strength']==grad]
                ax.plot(df_stats_conv_curr[col + '_conv bias[%]'], df_stats_conv_curr[col + '_conv std[%]'], color='k', markersize=10, marker='.', label='conv')
                ax.plot(df_stats_conv_curr[col + '_conv_nofilt bias[%]'], df_stats_conv_curr[col + '_conv_nofilt std[%]'], color='k', markersize=10, marker='*', label='conv nofilt')
                if legend_once:
                    legend_once = False
                    ax.legend()

        grid.fig.show()
        grid.fig.savefig(Path(analysis_results_dir) / f'{folder}_{col}{jitter_suffix}_{nb_realiz}seeds_biasstd_beta_perc.pdf')

# -----------------------------------------------------------------------------
# Compute bias-stddev in percentage

# the number of rows for each group should be = number of seeds/realizations
df_stats = df[params + ['recon_type','nofilt'] + criteria].copy()
ddf = df_stats.groupby(params + ['recon_type', 'nofilt'])
# compute the mean and stddev
df_stats = ddf.agg(['mean','std'])
# flatten the index multiindex for use with FacetGrid
df_stats = df_stats.reset_index()
# flatten the columns multiindex for use with FacetGrid
df_stats.columns = [' '.join(col).strip() for col in df_stats.columns.values]

# compute bias/std in perc and rename criteria
for i,c in enumerate(criteria):
        df_stats[c+' mean'] =  100 * (df_stats[c+' mean'].values - true[c]) / true[c]
        df_stats[c+' std'] =  100 * df_stats[c+' std'].values / true[c]
        df_stats = df_stats.rename(columns={c+' mean':c+' bias[%]', c+' std':c+' std[%]'})

df_stats = df_stats.rename(columns={'gradient_strength':'readout time'})
if load_nodecay:
    df_stats['readout time'] = df_stats['readout time'].cat.rename_categories(['x1','x0.7','x0.5', 'x1 no decay'])
else:
    df_stats['readout time'] = df_stats['readout time'].cat.rename_categories(['x1','x0.7','x0.5'])


for col in criteria:
    # show plots of bias-stddev in perc for comparing gradients
    grid = sns.relplot(
            data=df_stats, kind="line",
            x=col+' bias[%]', y=col+' std[%]', hue="readout time", style='recon_type',
            col="beta_gamma", markers={'agr':'.', 'conv':'*'}, markersize=20, legend='brief')

    for ax in grid.axes.ravel():
        ax.grid(ls=':')

    grid.fig.show()
    grid.fig.savefig(Path(analysis_results_dir) / f'{folder}_{col}{jitter_suffix}_{nb_realiz}seeds_biasstd_perc_grad.pdf')


# display bias-stddev over beta parameters
#df_stats = df[params + ['recon_type','nofilt'] + criteria].copy()
# normalize to perc and rename criteria
#for i,c in enumerate(criteria):
#        df_stats[c] =  100 * df_stats[c].values / true[c]
#        df_stats = df_stats.rename(columns={c:c+' [%]', c:c+' [%]'})
#for col in criteria:
#    # display bias-stddev over beta parameters
#    col_wrap = 4
#    grid = sns.catplot(data=df_stats, x="beta_recon", y=col, hue="gradient_strength", col='beta_gamma', kind="point", col_wrap=col_wrap, legend_out=False)
#    #grid.add_legend()
#    grid.fig.show()



# if loaded all the images, show the images of mean/stddev over realizations
#if load_all_images:
#    na = dict(cmap=plt.cm.viridis, vmin = 0, vmax = true_na_image.max())
#    prot = dict(cmap=plt.cm.gray)
#    ind_grad = df.gradient_strength.cat.categories.to_list().index("24")
#    agr_mean = np.mean(agr_na_realiz[ind_grad,0,0], axis=0)
#    agr_std = np.std(agr_na_realiz[ind_grad,0,0], axis=0)
#    vi = pv.ThreeAxisViewer([agr_mean, true_na_image, t1_image], imshow_kwargs=[na, na, prot])
#    gamma_mean = np.mean(gamma_na_realiz[ind_grad,0,0], axis=0)
#    gamma_std = np.std(gamma_na_realiz[ind_grad,0,0], axis=0)
#    vi1 = pv.ThreeAxisViewer([gamma_mean, gamma_std, true_na_image], imshow_kwargs=[prot, prot, na])

# mean comparison for checking
mask = gm_256 + wm_256 + csf_256
m_true = np.mean(true_na_image[mask])
for g in range(len(df.gradient_strength.cat.categories.to_list())):
    mi_agr = np.mean(agr_na_realiz[g, -1, -1][mask])
    m_conv = np.mean(conv_realiz[g][mask])
    print(f'{df.gradient_strength.cat.categories.to_list()[g]} example mean over brain: truth={m_true:.2g} agr={mi_agr:.2g} conv={m_conv:.2g}')

# mean over realiz
agr_na_realiz_mean /= nb_realiz


# script duration
duration = time.time() - start_time
print(f'took {duration}s')

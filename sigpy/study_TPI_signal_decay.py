""" Theoretical analysis of TPI T2* decay and SNR for different max gradients
"""

import numpy as np
from scipy.optimize import fmin_l_bfgs_b, least_squares
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import tpi_readout_time_from_k
import json
from pathlib import Path

#-------------------------------------------------------------------------------------


# bi-exponential decay over time
def biExpDecayOverTime(t2bs, t2bl, fracbl, concb, timepoints):
    return concb * ((1 - fracbl) * np.exp(-timepoints / t2bs) +
                    fracbl * np.exp(-timepoints / t2bl))


# monoexponential model
class mono_exp_model:
    def __init__(self, decay, timepoints):
        self.logdecay = np.log(decay)
        self.timepoints = timepoints

    def func(self, x):
        if len(x) != 1:
            raise Error
        return np.sum((np.divide(self.timepoints, self.logdecay) + x[0])**2)

    def grad(self, x):
        if len(x) != 1:
            raise Error
        return np.sum(2 * (np.divide(self.timepoints, self.logdecay) + x[0]))


# monoexponential fitting
def fit_mono_exp(timepoints, decay):
    current = mono_exp_model(decay, timepoints)
    x_r = np.ones(1)
    n = 100
    x_track = np.zeros(n)
    for j in range(n):
        res = least_squares(current.func, x_r)
        x_r = res.x
        #res = fmin_l_bfgs_b(current.func, x_r, current.grad)
        #x_r = res[0].copy()
        x_track[j] = x_r[0]

    plt.figure(), plt.plot(x_track, label='track'), plt.legend()
    return x_r[0]


#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    plt.ion()
    plt.rcParams.update({'font.size': 20})

    # T2* and Na concentration params
    t2bs = 3.
    t2bl = 20.
    t2fl = 50.
    t2fs = 52.
    fracbl = 0.4
    fracfl = 1.
    concb = 1.
    concf = 0.

    te = 0.45
    kmax = 1.45

    # read the data root directory from the config file
    with open('.simulation_config.json', 'r') as f:
        data_root_dir: str = json.load(f)['data_root_dir']

    # directory for saving results
    save_dir = Path(data_root_dir) / 'SNR_TPI_theoretical_analysis'
    save_dir.mkdir(exist_ok=True, parents=True)

    # pure T2* decay as a function of time
    timepoints = np.arange(0.4, 36., 0.1)
    # bi-exponential WM/GM tissue decay
    decay = biExpDecayOverTime(t2bs, t2bl, fracbl, concb, timepoints)
    # mono-exponential CSF decay
    csf_decay = np.exp(-timepoints / t2fl)

    # display pure T2* decays
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.tight_layout()
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('signal [a.u..]')
    ax.grid(visible=True)
    ax.plot(timepoints, decay, lw=5, label='GM, WM')
    ax.plot(timepoints, csf_decay, lw=5, label='CSF')
    plt.legend()
    plt.savefig(
        save_dir / f'decay_t_t2bs{t2bs:.0f}_l{t2bl:.0f}_t2fl{t2fl:.0f}.pdf')

    # T2* decay as a function of k magnitude for TPI (36ms readout)
    # k magnitude is linear, the corresponding time points are non linear
    num_pts = 32
    k = np.linspace(0., kmax, num_pts)
    timepoints = tpi_readout_time_from_k(k) + te
    decay = biExpDecayOverTime(t2bs, t2bl, fracbl, concb, timepoints)

    # fit mono exp decay to bi-exponential decay
    fitted_mono_decay_const = fit_mono_exp(timepoints, decay)
    #    fitted_mono_decay_const = 10. # ms
    mono_decay = np.exp(-timepoints / fitted_mono_decay_const)
    # display comparison bi exp - mono exp
    plt.figure()
    plt.plot(decay, label=f'bi {t2bs} + {t2bl}')
    plt.plot(mono_decay, label=f'mono {fitted_mono_decay_const:.1f}')
    plt.legend()

    # csf decay for different readout times (max gradients)
    csf_decay = np.exp(-timepoints / t2fl)
    csf_decay_faster_2 = np.exp(-(timepoints / 2.) / t2fl)
    csf_decay_faster_3 = np.exp(-(timepoints / 3.) / t2fl)
    csf_decay_faster_4 = np.exp(-(timepoints / 4.) / t2fl)
    csf_decay_faster_6 = np.exp(-(timepoints / 6.) / t2fl)

    # k magnitude signal decay for different readout times / max gradients
    decay_faster_2 = biExpDecayOverTime(t2bs, t2bl, fracbl, concb,
                                        timepoints / 2.)
    decay_faster_3 = biExpDecayOverTime(t2bs, t2bl, fracbl, concb,
                                        timepoints / 3.)
    decay_faster_4 = biExpDecayOverTime(t2bs, t2bl, fracbl, concb,
                                        timepoints / 4.)
    decay_faster_6 = biExpDecayOverTime(t2bs, t2bl, fracbl, concb,
                                        timepoints / 6.)

    # display signal decay as a function of k magnitude
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.tight_layout()
    ax.set_xlabel('|k| [1/cm]')
    ax.set_ylabel('decay factor')
    ax.set_ylim(0., 1.)
    ax.set_xlim(0., k.max())
    ax.grid(visible=True)
    ax.plot(k, decay, lw=5, label='WM')
    plt.plot(k, csf_decay, lw=5, label='CSF')
    plt.legend()
    plt.savefig(
        save_dir /
        f'decay_factor_k_t2bs{t2bs:.0f}_l{t2bl:.0f}_t2fl{t2fl:.0f}.pdf')

    # display signal decay as a function of k magnitude
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.tight_layout()
    ax.set_xlabel('|k| [1/cm]')
    ax.set_ylabel('decay factor')
    ax.set_ylim(0., 1.)
    ax.set_xlim(0., k.max())
    ax.grid(visible=True)
    ax.plot(k, decay, lw=5, label='0.16 G/cm')
    plt.plot(k, decay_faster_2, lw=5, label='0.32 G/cm')
    plt.plot(k, decay_faster_3, lw=5, label='0.48 G/cm')
    plt.plot(k, decay_faster_4, lw=5, label='0.64 G/cm')
    plt.plot(k, decay_faster_6, lw=5, label='0.96 G/cm')
    # frequency one before last on the 64 grid
    k_almost_max = k.max() - k.max() / 32
    #    ax.axvline(k_almost_max)
    plt.legend()
    plt.savefig(save_dir / f'decay_factor_k_t2bs{t2bs:.0f}_l{t2bl:.0f}.pdf')

    # display signal decay as a function of k magnitude
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.tight_layout()
    ax.set_xlabel('|k| [1/cm]')
    ax.set_ylabel('decay factor')
    ax.set_ylim(0., 1.)
    ax.set_xlim(0., k.max())
    ax.grid(visible=True)
    ax.plot(k, csf_decay, lw=5, label='0.16 G/cm')
    plt.plot(k, csf_decay_faster_2, lw=5, label='0.32 G/cm')
    plt.plot(k, csf_decay_faster_3, lw=5, label='0.48 G/cm')
    plt.plot(k, csf_decay_faster_4, lw=5, label='0.64 G/cm')
    plt.plot(k, csf_decay_faster_6, lw=5, label='0.96 G/cm')
    # frequency one before last on the 64 grid
    k_almost_max = k.max() - k.max() / 32
    #    ax.axvline(k_almost_max)
    plt.legend()
    plt.savefig(save_dir / f'decay_factor_k_t2fl{t2fl:.0f}.pdf')

    # theoretical factor that multiplies the original SNR (without decay) for each k-space frequency = signal decay / noise increase
    snr_factor_base = decay
    snr_factor_2 = decay_faster_2 / np.sqrt(2)
    snr_factor_3 = decay_faster_3 / np.sqrt(3)
    snr_factor_4 = decay_faster_4 / np.sqrt(4)
    snr_factor_6 = decay_faster_6 / np.sqrt(6)

    snr_factor_csf_base = csf_decay
    snr_factor_csf_2 = csf_decay_faster_2 / np.sqrt(2)
    snr_factor_csf_3 = csf_decay_faster_3 / np.sqrt(3)
    snr_factor_csf_4 = csf_decay_faster_4 / np.sqrt(4)
    snr_factor_csf_6 = csf_decay_faster_6 / np.sqrt(6)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.tight_layout()
    ax.set_xlabel('|k| [1/cm]')
    ax.set_ylabel('snr factor')
    ax.set_ylim(0., 1.)
    ax.set_xlim(0., k.max())
    ax.grid(visible=True)
    ax.plot(k, snr_factor_base, lw=5, label='0.16 G/cm')
    plt.plot(k, snr_factor_2, lw=5, label='0.32 G/cm')
    plt.plot(k, snr_factor_3, lw=5, label='0.48 G/cm')
    plt.plot(k, snr_factor_4, lw=5, label='0.64 G/cm')
    plt.plot(k, snr_factor_6, lw=5, label='0.96 G/cm')
    #    ax.axvline(k_almost_max)
    plt.legend()
    plt.savefig(save_dir / f'snr_factor_k_t2bs{t2bs:.0f}_l{t2bl:.0f}.pdf')

    # compare with mono decay : factor that multiplies the original SNR (without decay) for each k-space frequency = signal decay / noise increase
    snr_factor_base_mono = mono_decay
    mono_decay_2 = np.exp(-(timepoints / 2.) / fitted_mono_decay_const)
    snr_factor_2_mono = mono_decay_2 / np.sqrt(2.)
    mono_decay_3 = np.exp(-(timepoints / 3.) / fitted_mono_decay_const)
    snr_factor_3_mono = mono_decay_3 / np.sqrt(3.)
    mono_decay_4 = np.exp(-(timepoints / 4.) / fitted_mono_decay_const)
    snr_factor_4_mono = mono_decay_4 / np.sqrt(4.)
    mono_decay_6 = np.exp(-(timepoints / 6.) / fitted_mono_decay_const)
    snr_factor_6_mono = mono_decay_6 / np.sqrt(6.)

    # SNR factor for mono
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.tight_layout()
    ax.set_xlabel('|k| [1/cm]')
    ax.set_ylabel('snr factor')
    ax.set_ylim(0., 1.)
    ax.set_xlim(0., k.max())
    ax.grid(visible=True)
    ax.plot(k, snr_factor_base_mono, lw=5, label='mono 0.16 G/cm')
    plt.plot(k, snr_factor_2_mono, lw=5, label='mono 0.32 G/cm')
    plt.plot(k, snr_factor_3_mono, lw=5, label='mono 0.48 G/cm')
    plt.plot(k, snr_factor_4_mono, lw=5, label='mono 0.64 G/cm')
    plt.plot(k, snr_factor_6_mono, lw=5, label='mono 0.96 G/cm')
    plt.legend()
    plt.savefig(save_dir / f'snr_factor_k_t2bs{t2bs:.0f}_l{t2bl:.0f}_mono.pdf')

    # compare SNR factors for bi and mono for the lowest and highest maximum gradient
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.tight_layout()
    ax.set_xlabel('|k| [1/cm]')
    ax.set_ylabel('snr factor')
    ax.set_ylim(0., 1.)
    ax.set_xlim(0., k.max())
    ax.grid(visible=True)
    ax.plot(k, snr_factor_base, lw=5, label='0.16 G/cm')
    plt.plot(k, snr_factor_6, lw=5, label='0.96 G/cm')
    ax.plot(k, snr_factor_base_mono, lw=5, label='mono 0.16 G/cm')
    plt.plot(k, snr_factor_6_mono, lw=5, label='mono 0.96 G/cm')
    plt.legend()
    plt.savefig(
        save_dir / f'snr_factor_k_t2bs{t2bs:.0f}_l{t2bl:.0f}_bi_mono.pdf')

    # show the SNR gain with respect to slowest readout for white matter
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.tight_layout()
    ax.set_xlabel('|k| [1/cm]')
    ax.set_ylabel('WM SNR gain (|k|) ')
    ax.set_xlim(0., k.max())
    ax.grid(visible=True)
    plt.plot(k, snr_factor_base / snr_factor_base, lw=5, label='0.16 G/cm')
    plt.plot(k, snr_factor_2 / snr_factor_base, lw=5, label='0.32 G/cm')
    plt.plot(k, snr_factor_3 / snr_factor_base, lw=5, label='0.48 G/cm')
    plt.plot(k, snr_factor_4 / snr_factor_base, lw=5, label='0.64 G/cm')
    plt.plot(k, snr_factor_6 / snr_factor_base, lw=5, label='0.96 G/cm')
    #    ax.axvline(k_almost_max)
    plt.legend()
    plt.savefig(save_dir / f'snr_gain_k_t2bs{t2bs:.0f}_l{t2bl:.0f}.pdf')

    # show the SNR gain with respect to slowest readout for csf
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.tight_layout()
    ax.set_xlabel('|k| [1/cm]')
    ax.set_ylabel('snr factor')
    ax.set_xlim(0., k.max())
    ax.grid(visible=True)
    plt.plot(k, snr_factor_csf_base, lw=5, label='0.16 G/cm')
    plt.plot(k, snr_factor_csf_2, lw=5, label='0.32 G/cm')
    plt.plot(k, snr_factor_csf_3, lw=5, label='0.48 G/cm')
    plt.plot(k, snr_factor_csf_4, lw=5, label='0.64 G/cm')
    plt.plot(k, snr_factor_csf_6, lw=5, label='0.96 G/cm')
    #    ax.axvline(k_almost_max)
    plt.legend()
    plt.savefig(save_dir / f'snr_factor_k_t2fl{t2fl:.0f}.pdf')

    # show the SNR gain with respect to slowest readout for csf
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.tight_layout()
    ax.set_xlabel('|k| [1/cm]')
    ax.set_ylabel('CSF SNR gain (|k|)')
    ax.set_xlim(0., k.max())
    ax.grid(visible=True)
    plt.plot(
        k, snr_factor_csf_base / snr_factor_csf_base, lw=5, label='0.16 G/cm')
    plt.plot(
        k, snr_factor_csf_2 / snr_factor_csf_base, lw=5, label='0.32 G/cm')
    plt.plot(
        k, snr_factor_csf_3 / snr_factor_csf_base, lw=5, label='0.48 G/cm')
    plt.plot(
        k, snr_factor_csf_4 / snr_factor_csf_base, lw=5, label='0.64 G/cm')
    plt.plot(
        k, snr_factor_csf_6 / snr_factor_csf_base, lw=5, label='0.96 G/cm')
    #    ax.axvline(k_almost_max)
    plt.legend()
    plt.savefig(save_dir / f'snr_gain_k_t2fl{t2fl:.0f}.pdf')

    # SNR gain for the highest vs lowest gradient value, biexp vs monoexp
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.tight_layout()
    ax.set_xlabel('|k| [1/cm]')
    ax.set_ylabel('snr gain')
    ax.set_xlim(0., k.max())
    ax.grid(visible=True)
    plt.plot(k, snr_factor_base / snr_factor_base, lw=5, label='0.16 G/cm')
    plt.plot(k, snr_factor_6 / snr_factor_base, lw=5, label='0.96 G/cm')
    plt.plot(
        k,
        snr_factor_6_mono / snr_factor_base_mono,
        lw=5,
        label='mono 0.96 G/cm')
    plt.legend()
    plt.savefig(
        save_dir / f'snr_max_gain_k_t2bs{t2bs:.0f}_l{t2bl:.0f}_bi_mono.pdf')

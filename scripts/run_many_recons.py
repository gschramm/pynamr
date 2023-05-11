import argparse
import numpy as np
from numpy.random import default_rng
from pathlib import Path
import os
import sys

grad = [16, 24, 32]
folder = 'abstract_more'
#beta_gam = [3., 10., 30., 100.]
#beta_im = [0.1, 0.3, 1., 3.]
beta_gam = [10., 30.]
beta_im = [0.3, 1.]
#beta_gam = [3., 100.]
#beta_im = [0.1, 3.]

nb_realizations = 30 

rng = default_rng(42)

seeds = rng.integers(low=0, high=np.int64(1e7), size=nb_realizations)
seeds = seeds[1:2]

for s in seeds:
    for bg in beta_gam:
        for bi in beta_im:
            os.system(f'python reconstruct_simulated_dual_echo_tpi_na_data.py --gradient_strength=16 --noise_level=2e5 --beta_gamma={bg} --beta_recon={bi} --folder {folder} --seed {s} --no_decay')
            for g in grad:
                os.system(f'python reconstruct_simulated_dual_echo_tpi_na_data.py --gradient_strength={g} --noise_level=2e5 --beta_gamma={bg} --beta_recon={bi} --folder {folder} --seed {s}')


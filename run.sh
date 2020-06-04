#!/bin/bash

python multi_echo.py --noise_level 0.1 --nnearest 2 --nneigh 8  --bet_recon 3 --bet_gam 10
python multi_echo.py --noise_level 0.1 --nnearest 3 --nneigh 8  --bet_recon 3 --bet_gam 10
python multi_echo.py --noise_level 0.1 --nnearest 3 --nneigh 20 --bet_recon 3 --bet_gam 10

#!/bin/bash

python dual_echo.py --noise_level 0 --bet_gam 0.5 --bet_recon 0.5
python dual_echo.py --noise_level 0 --bet_gam 0.5 --bet_recon 1.0
python dual_echo.py --noise_level 0 --bet_gam 1.0 --bet_recon 0.5
python dual_echo.py --noise_level 0 --bet_gam 1.0 --bet_recon 1.0

python dual_echo.py --noise_level 0.1 --bet_gam 0.5 --bet_recon 0.5
python dual_echo.py --noise_level 0.1 --bet_gam 0.5 --bet_recon 1.0
python dual_echo.py --noise_level 0.1 --bet_gam 1.0 --bet_recon 0.5
python dual_echo.py --noise_level 0.1 --bet_gam 1.0 --bet_recon 1.0

python dual_echo.py --noise_level 0.2 --bet_gam 0.5 --bet_recon 0.5
python dual_echo.py --noise_level 0.2 --bet_gam 0.5 --bet_recon 1.0
python dual_echo.py --noise_level 0.2 --bet_gam 1.0 --bet_recon 0.5
python dual_echo.py --noise_level 0.2 --bet_gam 1.0 --bet_recon 1.0

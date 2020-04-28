#!/bin/bash

TLONG=-1
TSHORT=-1

python na_mr_recon_2d.py --n 256 --niter 500 --beta 0.3 --noise_level 0   --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 0.3 --noise_level 0.1 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 0.3 --noise_level 0.3 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG

python na_mr_recon_2d.py --n 256 --niter 500 --beta 1   --noise_level 0   --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 1   --noise_level 0.1 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 1   --noise_level 0.3 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG

python na_mr_recon_2d.py --n 256 --niter 500 --beta 3   --noise_level 0   --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 3   --noise_level 0.1 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 3   --noise_level 0.3 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG

TLONG=15
TSHORT=8

python na_mr_recon_2d.py --n 256 --niter 500 --beta 0.3 --noise_level 0   --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 0.3 --noise_level 0.1 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 0.3 --noise_level 0.3 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG

python na_mr_recon_2d.py --n 256 --niter 500 --beta 1   --noise_level 0   --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 1   --noise_level 0.1 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 1   --noise_level 0.3 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG

python na_mr_recon_2d.py --n 256 --niter 500 --beta 3   --noise_level 0   --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 3   --noise_level 0.1 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 3   --noise_level 0.3 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG

TLONG=50
TSHORT=50

python na_mr_recon_2d.py --n 256 --niter 500 --beta 0.3 --noise_level 0   --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 0.3 --noise_level 0.1 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 0.3 --noise_level 0.3 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG

python na_mr_recon_2d.py --n 256 --niter 500 --beta 1   --noise_level 0   --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 1   --noise_level 0.1 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 1   --noise_level 0.3 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG

python na_mr_recon_2d.py --n 256 --niter 500 --beta 3   --noise_level 0   --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 3   --noise_level 0.1 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 3   --noise_level 0.3 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG





#!/bin/bash

TLONG=-1
TSHORT=-1

python na_mr_recon_2d.py --n 256 --niter 500 --beta 3 --noise_level 0   --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 3 --noise_level 0.1 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 3 --noise_level 0.5 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG

python na_mr_recon_2d.py --n 256 --niter 500 --beta 10 --noise_level 0   --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 10 --noise_level 0.1 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 10 --noise_level 0.5 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG

python na_mr_recon_2d.py --n 256 --niter 500 --beta 30 --noise_level 0    --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 30 --noise_level 0.1 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 30 --noise_level 0.5 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG

TLONG=15
TSHORT=8

python na_mr_recon_2d.py --n 256 --niter 500 --beta 3 --noise_level 0   --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 3 --noise_level 0.1 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 3 --noise_level 0.5 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG

python na_mr_recon_2d.py --n 256 --niter 500 --beta 10 --noise_level 0   --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 10 --noise_level 0.1 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 10 --noise_level 0.5 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG

python na_mr_recon_2d.py --n 256 --niter 500 --beta 30 --noise_level 0   --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 30 --noise_level 0.1 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 30 --noise_level 0.5 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG

TLONG=50
TSHORT=50

python na_mr_recon_2d.py --n 256 --niter 500 --beta 3 --noise_level 0   --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 3 --noise_level 0.1 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 3 --noise_level 0.5 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG

python na_mr_recon_2d.py --n 256 --niter 500 --beta 10 --noise_level 0    --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 10 --noise_level 0.1 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 10 --noise_level 0.5 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG

python na_mr_recon_2d.py --n 256 --niter 500 --beta 30 --noise_level 0   --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 30 --noise_level 0.1 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG
python na_mr_recon_2d.py --n 256 --niter 500 --beta 30 --noise_level 0.5 --T2star_recon_short $TSHORT --T2star_recon_long $TLONG





from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import nibabel as nib

parser = ArgumentParser()
parser.add_argument('case')
parser.add_argument('--sdir', default = 'sodium_raw_data_kw0')
parser.add_argument('--n', default = 128, type = int)
args = parser.parse_args()

case = args.case
pdir = Path('data') / 'sodium_data' / 'birdcage' / args.case
sdir = args.sdir
n    = args.n

# create the output directory
odir = pdir / f'{sdir + "_preprocessed"}'
odir.mkdir(exist_ok = True)

# name of first echo file
fname = pdir / sdir / 'TE05.c0'
fname2 = pdir / sdir / 'TE5.c0'

#----
# load the complex coil image for first echo
data_shape  = (64,64,64)
recon_shape = (n,n,n) 

#---- load data of first echo
# load complex image
cimg = np.flip(np.fromfile(fname, dtype = np.complex64).reshape(data_shape).swapaxes(0,2), (1,2))
# perform fft to get into k-space
cimg_fft = np.fft.fftn(cimg, norm = 'ortho')
# pad data with 0s
data = np.fft.fftshift(np.pad(np.fft.fftshift(cimg_fft), (recon_shape[0] - data_shape[0])//2))

# load the complex coil image for 2nd echo
cimg2 = np.flip(np.fromfile(fname2, dtype = np.complex64).reshape(data_shape).swapaxes(0,2), (1,2))
# perform fft to get into k-space
cimg_fft2 = np.fft.fftn(cimg2, norm = 'ortho')
# pad data with 0s
data2 = np.fft.fftshift(np.pad(np.fft.fftshift(cimg_fft2), (recon_shape[0] - data_shape[0])//2))

#-------
data = np.expand_dims(data,0)
data2 = np.expand_dims(data2,0)

sens = np.ones_like(data)

# save the data and the sensitities
np.save(odir / f'echo1_{recon_shape[0]}.npy', data)
np.save(odir / f'echo2_{recon_shape[0]}.npy', data2)
np.save(odir / f'sens_{recon_shape[0]}.npy', sens)

#---------------------------------------------------------------------------------------------------
# load binary MR data

mr_path = list(pdir.glob('*MPRAGE*'))[0]
mr = np.flip(np.swapaxes(np.fromfile(mr_path, dtype = np.uint16).reshape(192,256,256), 0, 2), (0,1))

mr_dir = pdir / 'mprage_proton'
mr_dir.mkdir(exist_ok=True)

nib.save(nib.Nifti1Image(mr.astype(np.float32), np.eye(4)), str(mr_dir / 'T1.nii'))

#---------------------------------------------------------------------------------------------------
# visualizations

import matplotlib.pyplot as plt
import pymirc.viewer as pv
from scipy.ndimage import gaussian_filter

a = gaussian_filter(np.abs(cimg),1.2)
b = gaussian_filter(np.abs(cimg2),1.2)

vmax = np.percentile(a,99.99)

vi = pv.ThreeAxisViewer([a,b], imshow_kwargs = {'vmax': vmax, 'cmap': plt.cm.jet})
vi2 = pv.ThreeAxisViewer(mr)


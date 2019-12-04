# small demo script to verify implementation of discrete FT (with FFT)

import numpy as np
from   numba import njit, prange

import matplotlib.pyplot as py
py.ion()

#--------------------------------------------------------------

@njit(parallel = True)
def dft1D(f, k):
  n = f.shape[0]
  F = np.zeros(n, dtype = np.complex64)
  x = np.arange(n)

  for i in prange(n): 
    F[i] = (f*np.exp(-2*np.pi*k[i]*x*1j)).sum()

  return F

#--------------------------------------------------------------

@njit(parallel = True)
def idft1D(F, k):
  n = F.shape[0]
  f = np.zeros(n, dtype = np.complex64)
  x = np.arange(n)

  for i in prange(n): 
    f[i] = (F*np.exp(2*np.pi*k[i]*x*1j)).sum() / n

  return f

#--------------------------------------------------------------

@njit(parallel = True)
def apodized_dft1D(f, k, t, T2star):
  n = f.shape[0]
  F = np.zeros(n, dtype = np.complex64)
  x = np.arange(n)

  for i in prange(n): 
    E    = np.exp(-t[i]/T2star)
    F[i] = (E*f*np.exp(-2*np.pi*k[i]*x*1j)).sum()

  return F

#--------------------------------------------------------------

@njit(parallel = True)
def adjoint_apodized_dft1D(F, k, t, T2star):
  n = F.shape[0]
  f = np.zeros(n, dtype = np.complex64)
  x = np.arange(n)

  for i in prange(n): 
    E    = np.exp(-t/T2star[i])
    f[i] = (E*F*np.exp(2*np.pi*k[i]*x*1j)).sum()

  return f

#--------------------------------------------------------------

#--------------------------------------------------------------
#--------------------------------------------------------------
#--- set up the phantom
n = 256
x = np.arange(n)

f = np.zeros(n) + 0.34
f[(1*n//16):(3*n//16)]   = 1.1
f[(5*n//16):(7*n//16)]   = 0.7
f[(9*n//16):(11*n//16)]  = 0.7
f[(13*n//16):(15*n//16)] = 1.1
f[(15*n//16):] = 0
f[:(1*n//16)]  = 0

k = np.fft.fftfreq(n)
t = np.abs(k) * 400

T2star          = np.zeros(n) + 15
T2star[(1*n//16):(3*n//16)]   = 48
T2star[(5*n//16):(7*n//16)]   = 12
T2star[(9*n//16):(11*n//16)]  = 12
T2star[(13*n//16):(15*n//16)] = 48

signal = apodized_dft1D(f, k, t, T2star)

##-----------------------------------------------------------------
##-----------------------------------------------------------------
##--- power iterations to get the norm of the fwd operator
#
#b = np.random.rand(n) + np.random.rand(n)*1j
#
#for it in range(250):
#  b_fwd = apodized_dft1D(b, k, t, T2star)
#  norm  = np.sqrt((b_fwd * b_fwd.conj()).sum().real)
#  b     = b_fwd / norm
#  print(norm)

#-----------------------------------------------------------------
#-----------------------------------------------------------------
#--- do the recon

T2star_recon = T2star.copy()
#T2star_recon = np.zeros(n) + 48

# intitial recon is inverse fourier of signal
recon = idft1D(signal, k)

# do landweber update to get least square solution
niter = 500
step  = 1.9/n

recons = np.zeros((niter+1,n), dtype = np.complex)
recons[0,:] = recon

cost = np.zeros(niter)

for it in range(niter):
  exp_data = apodized_dft1D(recon, k, t, T2star_recon)
  diff     = exp_data - signal
  recon    = recon - step*adjoint_apodized_dft1D(diff, k, t, T2star_recon)
  recons[it + 1,:] = recon
  cost[it] = 0.5*(diff*diff.conj()).sum().real
  print(it + 1, niter, round(cost[it],4))

#--- make plots
fig, ax = py.subplots()
p1, = ax.plot(f,'k')
p2, = ax.plot(np.abs(recons[0,:]),'r')

for it in range(niter):
  p2.set_ydata(np.abs(recons[it+1,:]))
  py.pause(1e-6)
  ax.set_title('iteration ' + str(it) + ' cost ' + str(round(cost[it],4)))

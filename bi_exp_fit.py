# script to test influence of number of data points vs noise level for exp fitting

# start point matters (10*true_params -> bad fits)
# fitting mono-exp model to bi-exp data leads to bias in ampl when using many data points

# fitting bi-exp and low noise (0.3) -> more data better results
#                                    -> N = 2 low variance but bad fit in middle 

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import least_squares

def exp_model(t, params):
  """ mono / bi-exp decay model
  """
  if len(params) == 2:
    return params[0]*np.exp(-params[1]*t)
  elif len(params) == 3:
    return params[0]*(0.6*np.exp(-params[1]*t) + 0.4*np.exp(-params[2]*t))

def residuals(x, t, y):
  return (y - exp_model(t,x))

#----------------------------------------------

# true model parameters (amplitude, 1st decay const, 2nd decay const)
true_params = np.array([1., 1./8, 1./16])

T           = 10.                      # max acq time
noiselevel  = 1.*true_params[0]/20        # noise level (0.3 low, 1 high)
nreal       = 400                      # number of realizations
Ns          = np.array([2,3,4,8,32])   # number of acq. points
n_exp       = 1                        # number of exponents to fit (1 or 2)

#----------------------------------------------

np.random.seed(0)

print(f'true_params {true_params}\n')

tplot = np.linspace(0,T,100)
yplot = exp_model(tplot, true_params) 

x  = np.zeros((len(Ns),nreal,n_exp + 1))
x0 = 0.5*true_params[:(n_exp + 1)]

# array for plot prediction
p  = np.zeros((len(Ns),nreal,tplot.shape[0]))

A_bins     = np.linspace(0.5*true_params[0],1.5*true_params[0],51)
alpha_bins = np.linspace(-0.1,0.3,51)

fig, ax = plt.subplots(n_exp + 3, len(Ns), figsize = (9*Ns.shape[0]/4,9))

for k,N in enumerate(Ns):

  t = np.linspace(0,T,N)
  y = exp_model(t, true_params) 
  
  for i in range(nreal):
    #y_noise = y + noiselevel*np.random.randn(N)
    y_noise = y + noiselevel*np.sqrt(N)*np.random.randn(N)
    res = least_squares(residuals, x0, args = (t,y_noise))
  
    x[k,i,:] = res.x
    p[k,i,:] = exp_model(tplot, res.x)
  
  ax[0,k].plot(tplot,yplot)
  ax[0,k].plot(t,y_noise, '.--', color = 'tab:orange')
  ax[0,k].set_ylim(0, true_params[0]*1.3)

  ax[1,k].plot(A_bins[:-1], np.histogram(x[k,:,0], A_bins)[0], drawstyle = 'steps-post')

  for ie in range(n_exp):
    ax[2 + ie,k].plot(alpha_bins[:-1], np.histogram(x[k,:,1+ie], alpha_bins)[0], 
                      drawstyle = 'steps-post')
    if k == 0: ax[2+ie,0].set_ylabel(f'fitted alpha{ie+1} histogram')


  print(f'N {N}, A_mean     {x[k,:,0].mean():.5f}, A_std     {x[k,:,0].std():.5f}')
  for ie in range(n_exp):
    print(f'N {N}, alpha_mean {x[k,:,1+ie].mean():.5f}, alpha_std {x[k,:,1+ie].std():.5f}')
  print('')

  ax[0,k].set_title(f'N = {N}')
  ax[-1,k].fill_between(tplot, p[k].mean(0) + 2*p[k].std(0), p[k].mean(0) - 2*p[k].std(0), 
                        alpha = 0.5, color = 'tab:orange')
  ax[-1,k].plot(tplot, yplot, color = 'tab:blue')
  ax[-1,k].plot(tplot,p[k].mean(0), '--', color = 'tab:orange')
  ax[-1,k].set_ylim(0, true_params[0]*1.3)

ax[0,0].set_ylabel('data example + ground truth')
ax[1,0].set_ylabel('fitted A histogram')
ax[-1,0].set_ylabel('ground truth + fits')

for axx in ax.ravel(): axx.grid(ls = ':')
for axx in ax[-1,:].ravel(): axx.set_xlabel('t (ms)')

fig.tight_layout()
fig.show()

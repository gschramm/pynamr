import numpy as np

def Gamma(k, eta, c1, c2):
  return c1*np.exp(-eta*(k**3)) + c2

def readout_time(k, eta = 0.5387, c1 = 0.54, c2 = 0.46, 
                    p = 0.4, G = 0.16, gamma = 0.16, Km = 1.8):

  A = c2*(k**3) - (c1/eta)*np.exp(-eta*(k**3)) + (c1/eta)*np.exp(-eta*((p*Km)**3)) - c2*(p*Km)**3
  B = 3*Gamma(p*Km, eta, c1, c2)*(p**2)*(Km**3)
  C = Km/(gamma*G)

  return C*(p + A/B)

if __name__ == '__main__':
  import matplotlib.pyplot as py

  nk  = 64
  T2s = 30
  T2f = 8 

  rkwargs = {'eta'   : 0.5387, 
             'c1'    : 0.54, 
             'c2'    : 0.46, 
             'p'     : 0.4, 
             'G'     : 0.16, 
             'gamma' : 0.16, 
             'Km'    : 1.8}


  k = np.linspace(0, rkwargs['Km'], nk)
  t = readout_time(k, **rkwargs)

  decay_env = 0.6*np.exp(-t/T2f) + 0.4*np.exp(-t/T2s)

  fig, ax = py.subplots(1,3, figsize = (9,3))
  ax[0].plot(t,k,'.')
  ax[0].set_xlim((0,None))
  ax[0].axvline(44, color = 'k')

  ax[1].plot(k,t,'.')
  ax[1].axhline(44, color = 'k')

  ax[2].plot(k,decay_env,'.')

  ax[0].set_xlabel('t')
  ax[0].set_ylabel('k')
  ax[1].set_xlabel('k')
  ax[1].set_ylabel('t')
  ax[2].set_xlabel('k')
  ax[2].set_ylabel('decay env.')

  for axx in ax.flatten(): axx.grid(ls = ':')

  fig.suptitle(' '.join([x[0] + ':' + str(x[1]) for x in rkwargs.items()]), fontsize = 'small')
  fig.tight_layout(pad = 1.5)
  fig.show()


import numpy as np

def readout_time(k, 
                 lookup       = None,
                 eta          = 0.9830, 
                 c1           = 0.54, 
                 c2           = 0.46, 
                 alpha_sw_tpi = 18.95,
                 beta_sw_tpi  = -0.5171,
                 t0_sw        = 0.0018):

  if lookup is not None:
    tmp = np.loadtxt(lookup)
    x = tmp[:,1]
    y = tmp[:,0] / 1000

    t = np.interp(k,x,y)

  else:
    # the point until the readout is linear
    m = 1126 * 0.16

    k_lin = t0_sw * m

    i1 = np.where(k <= k_lin)
    i2 = np.where(k > k_lin)

    t = np.zeros(k.shape)
    t[i1] = k[i1] / m
    t[i2] = t0_sw + ((c2*(k[i2]**3) - ((c1/eta)*np.exp(-eta*(k[i2]**3))) - beta_sw_tpi) / (3*alpha_sw_tpi))

  return t

if __name__ == '__main__':
  import matplotlib.pyplot as py

  nk  = 128
  T2s = 0.029
  T2f = 0.009 
  osp = 1.8
  Km  = osp*0.8197

  rkwargs = {'lookup': 'kspace_times_4mm_g32.ksp'}

  #rkwargs = {'eta'          : 0.9830, 
  #           'c1'           : 0.54, 
  #           'c2'           : 0.46, 
  #           'alpha_sw_tpi' : 18.95,
  #           'beta_sw_tpi'  : -0.5171,
  #           't0_sw'        : 0.0018}
              
  k = np.linspace(0, Km, nk)
  t = readout_time(k, **rkwargs)

  decay_env = 0.6*np.exp(-t/T2f) + 0.4*np.exp(-t/T2s)

  fig, ax = py.subplots(1,4, figsize = (12,3))
  ax[0].plot(1000*t,k)
  ax[0].set_xlim((0,None))

  ax[1].plot(k,1000*t)
  #ax[2].axvline(rkwargs['t0_sw']*(1126*0.16), color = 'k', ls = ':')
  ax[2].plot(k,1000*t)
  ax[2].set_xlim((0,0.5))
  ax[2].set_ylim((0,3.5))

  ax[3].plot(k,decay_env)

  ax[0].set_xlabel('t (ms)')
  ax[0].set_ylabel('k')
  ax[1].set_xlabel('k')
  ax[1].set_ylabel('t( ms)')
  ax[2].set_xlabel('k (zoom)')
  ax[2].set_ylabel('t( ms)')
  ax[3].set_xlabel('k')
  ax[3].set_ylabel('decay env.')

  for axx in ax.flatten(): axx.grid(ls = ':')

  fig.suptitle(' '.join([x[0] + ':' + str(x[1]) for x in rkwargs.items()]), fontsize = 'small')
  fig.tight_layout(pad = 1.5)
  fig.show()


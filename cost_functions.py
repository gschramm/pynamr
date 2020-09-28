import cupy as cp

from apodized_fft import apodized_fft_multi_echo, adjoint_apodized_fft_multi_echo
from bowsher      import bowsher_prior_cost, bowsher_prior_grad

#--------------------------------------------------------------
def multi_echo_data_fidelity(recon, signal, readout_inds, Gam, tr, delta_t, nechos, kmask, sens):

  exp_data = apodized_fft_multi_echo(cp.asarray(recon), readout_inds, cp.asarray(Gam), tr, delta_t,
                                     cp.asarray(sens), nechos = nechos).get()
  diff     = (exp_data - signal)*kmask
  cost     = 0.5*(diff**2).sum()

  return cost

#--------------------------------------------------------------
def multi_echo_data_fidelity_grad(recon, signal, readout_inds, Gam, tr, delta_t, nechos, 
                                  kmask, grad_gamma, sens):

  exp_data = apodized_fft_multi_echo(cp.asarray(recon), readout_inds, cp.asarray(Gam), tr, delta_t, 
                                     cp.asarray(sens), nechos = nechos).get()
  diff     = (exp_data - signal)*kmask

  if grad_gamma:
    grad  = adjoint_apodized_fft_multi_echo(cp.asarray(diff), readout_inds, cp.asarray(Gam), tr, delta_t,
                                            cp.asarray(sens), grad_gamma = True).get()
    grad *= recon
  else:
    grad  = adjoint_apodized_fft_multi_echo(cp.asarray(diff), readout_inds, cp.asarray(Gam), tr, delta_t, 
                                            cp.asarray(sens), grad_gamma = False).get()

  return grad

#--------------------------------------------------------------------
def multi_echo_bowsher_cost(recon, recon_shape, signal, readout_inds, Gam, tr, delta_t, nechos, kmask,
                           beta, ninds, ninds2, method, sens):
  # ninds2 is a dummy argument to have the same arguments for
  # cost and its gradient

  isflat_recon = False
  isflat_Gam   = False

  if recon.ndim == 1:  
    isflat_recon = True
    recon  = recon.reshape(recon_shape)

  if Gam.ndim == 1:  
    isflat_Gam = True
    Gam  = Gam.reshape(recon_shape[:-1])

  cost = multi_echo_data_fidelity(recon, signal, readout_inds, Gam, tr, delta_t, nechos, kmask, sens)

  if beta > 0:
    cost += beta*bowsher_prior_cost(recon[...,0], ninds, method)
    cost += beta*bowsher_prior_cost(recon[...,1], ninds, method)

  if isflat_recon:
    recon = recon.flatten()

  if isflat_Gam:
    Gam = Gam.flatten()
   
  return cost

#--------------------------------------------------------------------
def multi_echo_bowsher_cost_gamma(Gam, recon_shape, signal, readout_inds, recon, tr, delta_t, nechos, kmask,
                           beta, ninds, ninds2, method, sens):
  # ninds2 is a dummy argument to have the same arguments for
  # cost and its gradient

  isflat_recon = False
  isflat_Gam   = False

  if recon.ndim == 1:  
    isflat_recon = True
    recon  = recon.reshape(recon_shape)

  if Gam.ndim == 1:  
    isflat_Gam = True
    Gam  = Gam.reshape(recon_shape[:-1])

  cost = multi_echo_data_fidelity(recon, signal, readout_inds, Gam, tr, delta_t, nechos, kmask, sens)

  if beta > 0:
    cost += beta*bowsher_prior_cost(Gam, ninds, method)

  if isflat_recon:
    recon = recon.flatten()

  if isflat_Gam:
    Gam = Gam.flatten()
   
  return cost

#--------------------------------------------------------------------
def multi_echo_bowsher_cost_total(recon, recon_shape, signal, readout_inds, Gam, tr, delta_t, nechos, kmask,
                             beta_recon, beta_gam, ninds, method, sens):
  # ninds2 is a dummy argument to have the same arguments for
  # cost and its gradient

  isflat_recon = False
  isflat_Gam   = False

  if recon.ndim == 1:  
    isflat_recon = True
    recon  = recon.reshape(recon_shape)

  if Gam.ndim == 1:  
    isflat_Gam = True
    Gam  = Gam.reshape(recon_shape[:-1])

  cost = multi_echo_data_fidelity(recon, signal, readout_inds, Gam, tr, delta_t, nechos, kmask, sens)

  if beta_recon > 0:
    cost += beta_recon*bowsher_prior_cost(recon[...,0], ninds, method)
    cost += beta_recon*bowsher_prior_cost(recon[...,1], ninds, method)

  if beta_gam > 0:
    cost += beta_gam*bowsher_prior_cost(Gam, ninds, method)

  if isflat_recon:
    recon = recon.flatten()

  if isflat_Gam:
    Gam = Gam.flatten()
   
  return cost


#--------------------------------------------------------------------
def multi_echo_bowsher_grad(recon, recon_shape, signal, readout_inds, Gam, tr, delta_t, nechos, kmask,
                           beta, ninds, ninds2, method, sens):

  isflat = False
  if recon.ndim == 1:  
    isflat = True
    recon  = recon.reshape(recon_shape)

  grad = multi_echo_data_fidelity_grad(recon, signal, readout_inds, Gam, tr, delta_t, 
                                       nechos, kmask, False, sens)

  if beta > 0:

    grad[...,0] += beta*bowsher_prior_grad(recon[...,0], ninds, ninds2, method)
    grad[...,1] += beta*bowsher_prior_grad(recon[...,1], ninds, ninds2, method)

  if isflat:
    recon = recon.flatten()
    grad  = grad.flatten()

  return grad

#--------------------------------------------------------------------
def multi_echo_bowsher_grad_gamma(Gam, recon_shape, signal, readout_inds, recon, tr, delta_t, nechos, 
                                  kmask, beta, ninds, ninds2, method, sens):

  isflat = False
  if Gam.ndim == 1:  
    isflat = True
    Gam = Gam.reshape(recon_shape[:-1])

  tmp = multi_echo_data_fidelity_grad(recon, signal, readout_inds, Gam, tr, delta_t, nechos, kmask, True, sens)

  grad = tmp[...,0] + tmp[...,1]

  if beta > 0:
    grad += beta*bowsher_prior_grad(Gam, ninds, ninds2, method)

  if isflat:
    Gam  = Gam.flatten()
    grad = grad.flatten()

  return grad

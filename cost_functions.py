from apodized_fft import apodized_fft, adjoint_apodized_fft
from bowsher      import bowsher_prior_cost, bowsher_prior_grad

#--------------------------------------------------------------
def mr_data_fidelity(recon, signal, readout_inds, apo_imgs, kmask):

  exp_data = apodized_fft(recon, readout_inds, apo_imgs)*kmask
  diff     = exp_data - signal
  cost     = 0.5*(diff**2).sum()

  return cost

#--------------------------------------------------------------
def mr_data_fidelity_grad(recon, signal, readout_inds, apo_imgs, kmask):

  exp_data = apodized_fft(recon, readout_inds, apo_imgs)*kmask
  diff     = exp_data - signal
  grad     = adjoint_apodized_fft(diff*kmask, readout_inds, apo_imgs)

  return grad

#--------------------------------------------------------------------
def mr_bowsher_cost(recon, recon_shape, signal, readout_inds, apo_imgs, 
                    beta, ninds, ninds2, method, kmask):
  # ninds2 is a dummy argument to have the same arguments for
  # cost and its gradient

  isflat = False
  if recon.ndim == 1:  
    isflat = True
    recon  = recon.reshape(recon_shape)

  cost = mr_data_fidelity(recon, signal, readout_inds, apo_imgs, kmask)

  if beta > 0:
    cost += beta*bowsher_prior_cost(recon[...,0], ninds, method)
    cost += beta*bowsher_prior_cost(recon[...,1], ninds, method)

  if isflat:
    recon = recon.flatten()
   
  return cost

#--------------------------------------------------------------------
def mr_bowsher_grad(recon, recon_shape, signal, readout_inds, apo_imgs, 
                    beta, ninds, ninds2, method, kmask):

  isflat = False
  if recon.ndim == 1:  
    isflat = True
    recon  = recon.reshape(recon_shape)

  grad = mr_data_fidelity_grad(recon, signal, readout_inds, apo_imgs, kmask)

  if beta > 0:

    grad[...,0] += beta*bowsher_prior_grad(recon[...,0], ninds, ninds2, method)
    grad[...,1] += beta*bowsher_prior_grad(recon[...,1], ninds, ninds2, method)

  if isflat:
    recon = recon.flatten()
    grad  = grad.flatten()

  return grad

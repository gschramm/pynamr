import numpy as np

def downsample4(a):
  a = a[::4,:,:] + a[1::4,:,:] + a[2::4,:,:] + a[3::4,:,:]
  a = a[:,::4,:] + a[:,1::4,:] + a[:,2::4,:] + a[:,3::4,:]
  a = a[:,:,::4] + a[:,:,1::4] + a[:,:,2::4] + a[:,:,3::4]

  return a / 64

def ellipse_inds(nx, ny, nz, rx, ry = None, rz = None, x0 = 0, y0 = 0, z0 = 0):

  if ry == None:
    ry = rx
  if rz == None:
    rz = rx

  x = np.arange(nx) - nx/2 + 0.5
  y = np.arange(ny) - ny/2 + 0.5
  z = np.arange(nz) - nz/2 + 0.5
  
  X,Y,Z = np.meshgrid(x,y,z, indexing = 'ij')
  
  return np.where((((X-x0)/rx)**2 + ((Y-y0)/ry)**2 + ((Z-z0)/rz)**2) <= 1)

#--------------------------------------------------------------
def ellipse_phantom(n = 128, cedge = 2, csph = 1.5, Rsp = None, gam_edge = 0.9, gam_center = 0.4):

  if Rsp is None: Rsp = n/12

  nphi = 6
  r    = n/4
  
  na  = np.zeros((n,n,n), dtype = np.float32)
  gam = np.ones((n,n,n), dtype = np.float32)
  t1  = np.zeros((n,n,n), dtype = np.float32)

  i0  = ellipse_inds(n, n, n,  n/2.2, ry = n/2.2, rz = n/4)
  i1  = ellipse_inds(n, n, n,  0.9*n/2.2, ry = 0.9*n/2.2, rz = 0.9*n/4)
  
  phis = np.linspace(0,2*np.pi,nphi+1)[:-1]
  
  na[i0] = cedge
  na[i1] = 1

  gam[i0] = gam_edge
  gam[i1] = gam_center

  t1[i0] = 0.5
  t1[i1] = 1.
  
  for i, phi in enumerate(phis):
    inds = ellipse_inds(n, n, n, Rsp, x0 = r*np.sin(phi), y0 = r*np.cos(phi))
    if (i % 2) == 0:
      na[inds] = csph
    if (i % 2) == 1:
      na[inds] = 1/csph
    if i < 2:
      t1[inds] = 1.3
    if (i >= 2) and (i < 4):
      inds2 = ellipse_inds(n, n, n, Rsp, x0 = r*np.sin(phi+np.pi/12), y0 = r*np.cos(phi+np.pi/12))
      t1[inds2] = 1.3

    inds2 = ellipse_inds(n, n, n, Rsp, x0 = 0, y0 = 0)
    t1[inds2] = 1.3

  t1 += 0.01*np.random.randn(*t1.shape)

  return na, gam, t1

if __name__ == '__main__':
  n   = 512
  Rsp = n/36

  a, b, c = ellipse_phantom(n, Rsp = Rsp)

  a = downsample4(a)
  b = downsample4(b)
  c = downsample4(c)


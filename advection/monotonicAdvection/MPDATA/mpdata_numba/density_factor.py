
from numba import njit, prange
from fvms.build_config import jitflags


def density_factor(cfg, psi, grgr, inverse=False):
  if inverse:
    __density_factor_inverse(psi, grgr, cfg.nx, cfg.ny, cfg.nz)
  else:
    __density_factor(psi, grgr, cfg.nx, cfg.ny, cfg.nz)

@njit(**jitflags)
def __density_factor(psi, grgr, nx, ny, nz):

  for jx in prange(0, nx): 
    for jy in prange(0, ny): 
      for jz in prange(0, nz):
        psi[jx,jy,jz] = psi[jx,jy,jz] * grgr[jx,jy,jz] 

@njit(**jitflags)
def __density_factor_inverse(psi, grgr, nx, ny, nz):

  for jx in prange(0, nx): 
    for jy in prange(0, ny): 
      for jz in prange(0, nz):
        psi[jx,jy,jz] = psi[jx,jy,jz] / grgr[jx,jy,jz] 
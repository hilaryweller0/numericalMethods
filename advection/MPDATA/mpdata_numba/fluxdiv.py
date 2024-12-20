
from numba import njit, prange
from fvms.build_config import jitflags


def fluxdiv(cfg, div, fxh, fyh, fzh):
  __fluxdiv(div, fxh, fyh, fzh, cfg.nx, cfg.ny, cfg.nz, cfg.dx, cfg.dy, cfg.dz)

@njit(**jitflags)
def __fluxdiv(div, fxh, fyh, fzh, nx, ny, nz, dx, dy, dz):

  dxi = 1.0 / dx
  dyi = 1.0 / dy
  dzi = 1.0 / dz

  for jx in prange(0, nx): 
    for jy in prange(0, ny): 
      for jz in prange(0, nz):
        div[jx,jy,jz] = (fxh[jx+1, jy , jz ] - fxh[jx,jy,jz]) * dxi \
                      + (fyh[ jx ,jy+1, jz ] - fyh[jx,jy,jz]) * dyi \
                      + (fzh[ jx , jy ,jz+1] - fzh[jx,jy,jz]) * dzi
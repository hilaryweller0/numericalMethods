
import numpy as np
from numba import njit, prange, stencil, config
from itertools import product
from fvms.build_config import jitflags

#print("NUMTHREADS:", config.NUMBA_DEFAULT_NUM_THREADS)

def update_fluxdiv(cfg, psi, grg, fxh, fyh, fzh):

  __update_fluxdiv(psi, grg, fxh, fyh, fzh, cfg.nx, cfg.ny, cfg.nz, cfg.dx, cfg.dy, cfg.dz, cfg.dt)


@njit(**jitflags)
def __update_fluxdiv(psi, grg, fxh, fyh, fzh, nx, ny, nz, dx, dy, dz, dt):

  dxi = 1.0 / dx
  dyi = 1.0 / dy
  dzi = 1.0 / dz

  for jx in prange(0, nx): 
    for jy in prange(0, ny): 
      for jz in prange(0, nz):
        div =   (fxh[jx+1, jy , jz ] - fxh[jx,jy,jz]) * dxi \
              + (fyh[ jx ,jy+1, jz ] - fyh[jx,jy,jz]) * dyi \
              + (fzh[ jx , jy ,jz+1] - fzh[jx,jy,jz]) * dzi
        psi[jx,jy,jz] = psi[jx,jy,jz] - dt * div / grg[jx,jy,jz]


#@njit # numba doesn't work with itertools product...
def update_fluxdiv_itertools(cfg, psi, grg, fxh, fyh, fzh):

  dxi = 1.0 / cfg.dx
  dyi = 1.0 / cfg.dy
  dzi = 1.0 / cfg.dz

  nx = cfg.nx
  ny = cfg.ny
  nz = cfg.nz

  for jx, jy, jz in product(range(0, nx), range(0, ny), range(0, nz)): 
    div =   (fxh[jx+1, jy , jz ] - fxh[jx,jy,jz]) * dxi \
          + (fyh[ jx ,jy+1, jz ] - fyh[jx,jy,jz]) * dyi \
          + (fzh[ jx , jy ,jz+1] - fzh[jx,jy,jz]) * dzi
    psi[jx,jy,jz] = psi[jx,jy,jz]-cfg.dt*div/grg[jx,jy,jz]


#@njit
def update_fluxdiv_vec(cfg, psi, grg, fxh, fyh, fzh):

    dxi = 1.0 / cfg.dx
    dyi = 1.0 / cfg.dy
    dzi = 1.0 / cfg.dz

    div = np.empty_like(psi)

    nx = cfg.nx
    ny = cfg.ny
    nz = cfg.nz

    div[0:nx,0:ny,0:nz] = (fxh[1:nx+1,0:ny,0:nz] - fxh[0:nx,0:ny,0:nz]) * dxi \
                        + (fyh[0:nx,1:ny+1,0:nz] - fyh[0:nx,0:ny,0:nz]) * dyi \
                        + (fzh[0:nx,0:ny,1:nz+1] - fzh[0:nx,0:ny,0:nz]) * dzi
    psi[0:nx,0:ny,0:nz] = psi[0:nx,0:ny,0:nz]-cfg.dt*div[0:nx,0:ny,0:nz]/grg[0:nx,0:ny,0:nz]
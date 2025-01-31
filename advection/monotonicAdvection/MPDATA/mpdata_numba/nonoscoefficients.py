
from numba import njit, prange
from fvms.build_config import jitflags
from fvms.mpdata.common import _fneg, _fpos

def nonoscoefficients(cfg, psimin, psimax, fxh, fyh, fzh, grg, psi, cnh, cnl):

  __nonoscoefficients(psimin, psimax, fxh, fyh, fzh, grg, psi, cnh, cnl,
                      cfg.nx, cfg.ny, cfg.nz, cfg.dx, cfg.dy, cfg.dz, cfg.dt)

@njit(**jitflags)
def __nonoscoefficients(psimin, psimax, fxh, fyh, fzh, grg, psi, cnh, cnl,
                      nx, ny, nz, dx, dy, dz, dt):

  dxi = 1.0 / dx
  dyi = 1.0 / dy
  dzi = 1.0 / dz
  dti = 1.0 / dt
  eps = 1.0e-15

  for jx in prange(0, nx): 
    for jy in prange(0, ny): 
      for jz in prange(0, nz):

        fluxin  =   (- _fneg(fxh[jx+1, jy , jz ]) + _fpos(fxh[jx,jy,jz])) * dxi \
                  + (- _fneg(fyh[ jx ,jy+1, jz ]) + _fpos(fyh[jx,jy,jz])) * dyi \
                  + (- _fneg(fzh[ jx , jy ,jz+1]) + _fpos(fzh[jx,jy,jz])) * dzi
        fluxout =   (  _fpos(fxh[jx+1, jy , jz ]) - _fneg(fxh[jx,jy,jz])) * dxi \
                  + (  _fpos(fyh[ jx ,jy+1, jz ]) - _fneg(fyh[jx,jy,jz])) * dyi \
                  + (  _fpos(fzh[ jx , jy ,jz+1]) - _fneg(fzh[jx,jy,jz])) * dzi

        cnh[jx,jy,jz] = (psimax[jx,jy,jz] - psi[jx,jy,jz]) * grg[jx,jy,jz] * dti / (fluxin + eps)
        cnl[jx,jy,jz] = (psi[jx,jy,jz] - psimin[jx,jy,jz]) * grg[jx,jy,jz] * dti / (fluxout + eps) 
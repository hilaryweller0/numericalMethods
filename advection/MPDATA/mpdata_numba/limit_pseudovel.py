import numpy as np
from numba import njit, prange
from fvms.build_config import jitflags
from fvms.mpdata.common import _fneg, _fpos

def limit_pseudovel(cfg, cnh, cnl, fxh, fyh, fzh, vxh, vyh, vzh): 
  limit_pseudovelx(cnh, cnl, fxh, vxh, cfg.nx, cfg.ny, cfg.nz, cfg.bcx)
  limit_pseudovely(cnh, cnl, fyh, vyh, cfg.nx, cfg.ny, cfg.nz, cfg.bcy)
  limit_pseudovelz(cnh, cnl, fzh, vzh, cfg.nx, cfg.ny, cfg.nz, cfg.bcz)

@njit(**jitflags)
def limit_pseudovelx(cnh, cnl, fxh, vxh, nx, ny, nz, bcx):

  for jx in prange(1, nx): 
    for jy in prange(0, ny): 
      for jz in prange(0, nz):
        vxh[jx,jy,jz] =   _fpos(fxh[jx,jy,jz]) * min(1.0, cnh[ jx ,jy,jz], cnl[jx-1,jy,jz]) \
                        + _fneg(fxh[jx,jy,jz]) * min(1.0, cnh[jx-1,jy,jz], cnl[ jx ,jy,jz])

  # no limiter for pseudovel on open boundaries                       
  if bcx == 1:
    for jy in prange(0, ny): 
      for jz in prange(0, nz):
        vxh[ 0,jy,jz] =   _fpos(fxh[ 0,jy,jz]) * min(1.0, cnh[ 0,jy,jz], cnl[-2,jy,jz]) \
                        + _fneg(fxh[ 0,jy,jz]) * min(1.0, cnh[-2,jy,jz], cnl[ 0,jy,jz]) 
        vxh[-1,jy,jz] =   _fpos(fxh[-1,jy,jz]) * min(1.0, cnh[ 1,jy,jz], cnl[-1,jy,jz]) \
                        + _fneg(fxh[-1,jy,jz]) * min(1.0, cnh[-1,jy,jz], cnl[ 1,jy,jz]) 



@njit(**jitflags)
def limit_pseudovely(cnh, cnl, fyh, vyh, nx, ny, nz, bcy):

  for jx in prange(0, nx): 
    for jy in prange(1, ny): 
      for jz in prange(0, nz):
        vyh[jx,jy,jz] =   _fpos(fyh[jx,jy,jz]) * min(1.0, cnh[jx, jy ,jz], cnl[jx,jy-1,jz]) \
                        + _fneg(fyh[jx,jy,jz]) * min(1.0, cnh[jx,jy-1,jz], cnl[jx, jy ,jz])

  # no limiter for pseudovel on open boundaries                       
  if bcy == 1:
    for jx in prange(0, nx): 
      for jz in prange(0, nz):
        vyh[jx, 0,jz] =   _fpos(fyh[jx, 0,jz]) * min(1.0, cnh[jx, 0,jz], cnl[jx,-2,jz]) \
                        + _fneg(fyh[jx, 0,jz]) * min(1.0, cnh[jx,-2,jz], cnl[jx, 0,jz]) 
        vyh[jx,-1,jz] =   _fpos(fyh[jx,-1,jz]) * min(1.0, cnh[jx, 1,jz], cnl[jx,-1,jz]) \
                        + _fneg(fyh[jx,-1,jz]) * min(1.0, cnh[jx,-1,jz], cnl[jx, 1,jz]) 


@njit(**jitflags)
def limit_pseudovelz(cnh, cnl, fzh, vzh, nx, ny, nz, bcz):

  for jx in prange(0, nx): 
    for jy in prange(0, ny): 
      for jz in prange(1, nz):
        vzh[jx,jy,jz] =   _fpos(fzh[jx,jy,jz]) * min(1.0, cnh[jx,jy, jz ], cnl[jx,jy,jz-1]) \
                        + _fneg(fzh[jx,jy,jz]) * min(1.0, cnh[jx,jy,jz-1], cnl[jx,jy, jz ])

  # no limiter for pseudovel on open boundaries                       
  if bcz == 1:
    for jx in prange(0, nx): 
      for jy in prange(0, ny):
        vzh[jx,jy, 0] =   _fpos(fzh[jx,jy, 0]) * min(1.0, cnh[jx,jy, 0], cnl[jx,jy,-2]) \
                        + _fneg(fzh[jx,jy, 0]) * min(1.0, cnh[jx,jy,-2], cnl[jx,jy, 0]) 
        vzh[jx,jy,-1] =   _fpos(fzh[jx,jy,-1]) * min(1.0, cnh[jx,jy, 1], cnl[jx,jy,-1]) \
                        + _fneg(fzh[jx,jy,-1]) * min(1.0, cnh[jx,jy,-1], cnl[jx,jy, 1]) 
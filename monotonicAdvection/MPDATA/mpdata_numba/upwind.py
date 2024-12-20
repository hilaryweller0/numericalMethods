
import numpy as np
from numba import njit, prange
from fvms.build_config import jitflags

def upwind(cfg, psi, vxh, vyh, vzh, fxh, fyh, fzh):

  upwindx(psi, vxh, fxh, cfg.nx, cfg.ny, cfg.nz, cfg.bcx)
  upwindy(psi, vyh, fyh, cfg.nx, cfg.ny, cfg.nz, cfg.bcy)
  upwindz(psi, vzh, fzh, cfg.nx, cfg.ny, cfg.nz, cfg.bcz)


@njit(**jitflags)
def upwindx(psi, vxh, fxh, nx, ny, nz, bcx):

  for jx in prange(1, nx): 
    for jy in prange(0, ny): 
      for jz in prange(0, nz):
        fxh[jx,jy,jz] = np.maximum(0.0, vxh[jx,jy,jz]) * psi[jx-1,jy,jz] \
                      + np.minimum(0.0, vxh[jx,jy,jz]) * psi[ jx ,jy,jz]  
    
  if bcx == 0:
    for jy in prange(0, ny): 
      for jz in prange(0, nz):
        fxh[ 0,jy,jz] = - fxh[ 1,jy,jz]
        fxh[-1,jy,jz] = - fxh[-2,jy,jz]

  elif bcx == 1:
    for jy in prange(0, ny): 
      for jz in prange(0, nz):
        fxh[ 0,jy,jz] = fxh[-2,jy,jz]
        fxh[-1,jy,jz] = fxh[ 1,jy,jz]



@njit(**jitflags)
def upwindy(psi, vyh, fyh, nx, ny, nz, bcy):
    
  for jx in prange(0, nx): 
    for jy in prange(1, ny): 
      for jz in prange(0, nz):
        fyh[jx,jy,jz] = np.maximum(0.0, vyh[jx,jy,jz]) * psi[jx,jy-1,jz] \
                      + np.minimum(0.0, vyh[jx,jy,jz]) * psi[jx, jy ,jz]  
    
  if bcy == 0:
    for jx in prange(0, nx): 
      for jz in prange(0, nz):
        fyh[jx, 0,jz] = - fyh[jx, 1,jz]
        fyh[jx,-1,jz] = - fyh[jx,-2,jz]
      
  elif bcy == 1:
    for jx in prange(0, nx): 
      for jz in prange(0, nz):
        fyh[jx, 0,jz] = fyh[jx,-2,jz]
        fyh[jx,-1,jz] = fyh[jx, 1,jz]



@njit(**jitflags)
def upwindz(psi, vzh, fzh, nx, ny, nz, bcz):

  for jx in prange(0, nx): 
    for jy in prange(0, ny): 
      for jz in prange(1, nz):
        fzh[jx,jy,jz] = np.maximum(0.0, vzh[jx,jy,jz]) * psi[jx,jy,jz-1] \
                      + np.minimum(0.0, vzh[jx,jy,jz]) * psi[jx,jy, jz ]  
    
  if bcz == 0:
    for jx in prange(0, nx): 
      for jy in prange(0, ny):
        fzh[jx,jy, 0] = - fzh[jx,jy, 1]
        fzh[jx,jy,-1] = - fzh[jx,jy,-2]
      
    
  elif bcz == 1:
    for jx in prange(0, nx): 
      for jy in prange(0, ny):
         fzh[jx,jy, 0] = fzh[jx,jy,-2]
         fzh[jx,jy,-1] = fzh[jx,jy, 1]

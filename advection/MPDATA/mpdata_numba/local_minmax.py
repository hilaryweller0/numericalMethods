
import numpy as np
from numba import njit, prange, stencil, config
from itertools import product
from fvms.build_config import jitflags

#print("NUMTHREADS:", config.NUMBA_DEFAULT_NUM_THREADS)

def local_minmax(cfg, psi, psimin, psimax, linit_minmax=False):

  if linit_minmax:
    local_minmax_init(psi, psimin, psimax, cfg.nx, cfg.ny, cfg.nz)

  local_minmax_x(psi, psimin, psimax, cfg.nx, cfg.ny, cfg.nz, cfg.bcx)
  local_minmax_y(psi, psimin, psimax, cfg.nx, cfg.ny, cfg.nz, cfg.bcy)
  local_minmax_z(psi, psimin, psimax, cfg.nx, cfg.ny, cfg.nz, cfg.bcz)


@njit(**jitflags)
def local_minmax_init(psi, psimin, psimax, nx, ny, nz):

  for jx in prange(0, nx): 
    for jy in prange(0, ny): 
      for jz in prange(0, nz):
        psimin[jx,jy,jz] = psi[jx,jy,jz]
        psimax[jx,jy,jz] = psi[jx,jy,jz]

@njit(**jitflags)
def local_minmax_x(psi, psimin, psimax, nx, ny, nz, bcx):

  for jx in prange(1, nx-1): 
    for jy in prange(0, ny): 
      for jz in prange(0, nz):
        psimin[jx,jy,jz] = min(psi[jx-1,jy,jz], psimin[jx,jy,jz], psi[jx+1,jy,jz]) 
        psimax[jx,jy,jz] = max(psi[jx-1,jy,jz], psimax[jx,jy,jz], psi[jx+1,jy,jz]) 
  
  if bcx == 0: 
    for jy in prange(0, ny): 
      for jz in prange(0, nz):
        psimin[ 0,jy,jz] = min(psimin[ 0,jy,jz], psi[ 1,jy,jz]) 
        psimax[ 0,jy,jz] = max(psimax[ 0,jy,jz], psi[ 1,jy,jz]) 
        psimin[-1,jy,jz] = min(psi[-2,jy,jz], psimin[-1,jy,jz]) 
        psimax[-1,jy,jz] = max(psi[-2,jy,jz], psimax[-1,jy,jz]) 

  if bcx == 1: 
    for jy in prange(0, ny): 
      for jz in prange(0, nz):
        psimin[ 0,jy,jz] = min(psi[-2,jy,jz], psimin[ 0,jy,jz], psi[ 1,jy,jz]) 
        psimax[ 0,jy,jz] = max(psi[-2,jy,jz], psimax[ 0,jy,jz], psi[ 1,jy,jz]) 
        psimin[-1,jy,jz] = min(psi[-2,jy,jz], psimin[-1,jy,jz], psi[ 1,jy,jz]) 
        psimax[-1,jy,jz] = max(psi[-2,jy,jz], psimax[-1,jy,jz], psi[ 1,jy,jz]) 


@njit(**jitflags)
def local_minmax_y(psi, psimin, psimax, nx, ny, nz, bcy):

  for jx in prange(0, nx): 
    for jy in prange(1, ny-1): 
      for jz in prange(0, nz):
        psimin[jx,jy,jz] = min(psi[jx,jy-1,jz], psimin[jx,jy,jz], psi[jx,jy+1,jz]) 
        psimax[jx,jy,jz] = max(psi[jx,jy-1,jz], psimax[jx,jy,jz], psi[jx,jy+1,jz]) 


  if bcy == 0: 
    for jx in prange(0, nx): 
      for jz in prange(0, nz):
        psimin[jx, 0,jz] = min(psimin[jx, 0,jz], psi[jx, 1,jz]) 
        psimax[jx, 0,jz] = max(psimax[jx, 0,jz], psi[jx, 1,jz]) 
        psimin[jx,-1,jz] = min(psi[jx,-2,jz], psimin[jx,-1,jz]) 
        psimax[jx,-1,jz] = max(psi[jx,-2,jz], psimax[jx,-1,jz]) 

  if bcy == 1: 
    for jx in prange(0, nx): 
      for jz in prange(0, nz):
        psimin[jx, 0,jz] = min(psi[jx,-2,jz], psimin[jx, 0,jz], psi[jx, 1,jz]) 
        psimax[jx, 0,jz] = max(psi[jx,-2,jz], psimax[jx, 0,jz], psi[jx, 1,jz]) 
        psimin[jx,-1,jz] = min(psi[jx,-2,jz], psimin[jx,-1,jz], psi[jx, 1,jz]) 
        psimax[jx,-1,jz] = max(psi[jx,-2,jz], psimax[jx,-1,jz], psi[jx, 1,jz]) 


@njit(**jitflags)
def local_minmax_z(psi, psimin, psimax, nx, ny, nz, bcz):

  for jx in prange(0, nx): 
    for jy in prange(0, ny): 
      for jz in prange(1, nz-1):
        psimin[jx,jy,jz] = min(psi[jx,jy,jz-1], psimin[jx,jy,jz], psi[jx,jy,jz+1]) 
        psimax[jx,jy,jz] = max(psi[jx,jy,jz-1], psimax[jx,jy,jz], psi[jx,jy,jz+1]) 

  if bcz == 0: 
    for jx in prange(0, nx): 
      for jy in prange(0, ny):
        psimin[jx,jy, 0] = min(psimin[jx,jy, 0], psi[jx,jy, 1]) 
        psimax[jx,jy, 0] = max(psimax[jx,jy, 0], psi[jx,jy, 1]) 
        psimin[jx,jy,-1] = min(psi[jx,jy,-2], psimin[jx,jy,-1]) 
        psimax[jx,jy,-1] = max(psi[jx,jy,-2], psimax[jx,jy,-1]) 

  if bcz == 1: 
    for jx in prange(0, nx): 
      for jy in prange(0, ny):
        psimin[jx,jy, 0] = min(psi[jx,jy,-2], psimin[jx,jy, 0], psi[jx,jy, 1]) 
        psimax[jx,jy, 0] = max(psi[jx,jy,-2], psimax[jx,jy, 0], psi[jx,jy, 1]) 
        psimin[jx,jy,-1] = min(psi[jx,jy,-2], psimin[jx,jy,-1], psi[jx,jy, 1]) 
        psimax[jx,jy,-1] = max(psi[jx,jy,-2], psimax[jx,jy,-1], psi[jx,jy, 1]) 

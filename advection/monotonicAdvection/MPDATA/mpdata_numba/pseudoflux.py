import numpy as np
from numba import njit, prange
from fvms.build_config import jitflags


def pseudo_flux(cfg, psi, grg, div, vxh, vyh, vzh, fxh, fyh, fzh): 
  __pseudo_flux(psi, grg, div, vxh, fxh, cfg.nx, cfg.ny, cfg.nz, cfg.bcx, cfg.dt)
  __pseudo_fluy(psi, grg, div, vyh, fyh, cfg.nx, cfg.ny, cfg.nz, cfg.bcy, cfg.dt)
  __pseudo_fluz(psi, grg, div, vzh, fzh, cfg.nx, cfg.ny, cfg.nz, cfg.bcz, cfg.dt)


@njit(**jitflags)
def __pseudo_flux(psi, grg, div, vxh, fxh, nx, ny, nz, bcx, dt):

  for jx in prange(1, nx): 
    for jy in prange(0, ny): 
      for jz in prange(0, nz):
        vxt = 0.5 * np.abs(vxh[jx,jy,jz]) * (psi[jx,jy,jz] - psi[jx-1,jy,jz])
        grgi = 1.0 / (grg[jx,jy,jz] + grg[jx-1,jy,jz])
        vxc = - 0.5 * dt * vxh[jx,jy,jz] * grgi * (div[jx,jy,jz] + div[jx-1,jy,jz])
        fxh[jx,jy,jz] = vxt + vxc 

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
def __pseudo_fluy(psi, grg, div, vyh, fyh, nx, ny, nz, bcy, dt):

  for jx in prange(0, nx): 
    for jy in prange(1, ny): 
      for jz in prange(0, nz):
        vyt = 0.5 * np.abs(vyh[jx,jy,jz]) * (psi[jx,jy,jz] - psi[jx,jy-1,jz])
        grgi = 1.0 / (grg[jx,jy,jz] + grg[jx,jy-1,jz])
        vyc = - 0.5 * dt * vyh[jx,jy,jz] * grgi * (div[jx,jy,jz] + div[jx,jy-1,jz])
        fyh[jx,jy,jz] = vyt + vyc 

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
def __pseudo_fluz(psi, grg, div, vzh, fzh, nx, ny, nz, bcz, dt):

  for jx in prange(0, nx): 
    for jy in prange(0, ny): 
      for jz in prange(1, nz):
        vzt = 0.5 * np.abs(vzh[jx,jy,jz]) * (psi[jx,jy,jz] - psi[jx,jy,jz-1])
        grgi = 1.0 / (grg[jx,jy,jz] + grg[jx,jy,jz-1])
        vzc = - 0.5 * dt * vzh[jx,jy,jz] * grgi * (div[jx,jy,jz] + div[jx,jy,jz-1])
        fzh[jx,jy,jz] = vzt + vzc 

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



@njit(**jitflags)
def pseudovelxy_adv(psi, grg, vxh, fxh):

  #for jx in prange(1, nx): 
  #  for jy in prange(0, ny): 
  #    for jz in prange(0, nz):
  #      grgi = 1.0 / (grg[jx,jy,jz] + grg[jx-1,jy,jz])
  #      vxc = 0.5 * np.abs(vxh[jx,jy,jz]) - vxh[jx,jy,jz]**2 * dt_dx * grgi  
  #      vyfx = 0.25 * (vyh[jx,jy,jz] + vyh[jx,jy,jz] + vyh[jx,jy,jz] + vyh[jx,jy,jz])
  #      vyc = - vxh[jx,jy,jz] * vyh[jx,jy,jz] * dt_dy * grgi  
  #      fxh[jx,jy,jz] = vxc * (psi[jx,jy,jz] - psi[jx-1,jy,jz])

  dt_dx = dt/dx

  for jx in prange(1, nx): 
    for jy in prange(0, ny): 
      for jz in prange(0, nz):
        grgi = 1.0 / (grg[jx,jy,jz] + grg[jx-1,jy,jz])
        vxc = 0.5 * np.abs(vxh[jx,jy,jz]) - vxh[jx,jy,jz]**2 * dt_dx * grgi  
        #vyc = - vxh[jx,jy,jz] * dt_dy * grgi  
        fxh[jx,jy,jz] = vxc * (psi[jx,jy,jz] - psi[jx-1,jy,jz])

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

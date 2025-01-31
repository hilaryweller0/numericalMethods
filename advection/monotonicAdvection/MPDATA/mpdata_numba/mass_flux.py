
from numba import njit
from fvms.build_config import jitflags


@njit(**jitflags)
def _fpos(f):
    return max(0.0, f)

@njit(**jitflags)
def _fneg(f):
    return min(0.0, f)


def mass_flux_upwind(cfg, axh, ayh, azh, bxh, byh, bzh): 
  __mass_flux_upwind(axh, ayh, azh, bxh, byh, bzh)

@njit(**jitflags)
def __mass_flux_upwind(axh, ayh, azh, bxh, byh, bzh):

  bxh[:,:,:] = axh[:,:,:]
  byh[:,:,:] = ayh[:,:,:] 
  bzh[:,:,:] = azh[:,:,:]


def mass_flux_accumulated(cfg, axh, ayh, azh, bxh, byh, bzh, cxh, cyh, czh): 
  __mass_flux_accumulated(axh, ayh, azh, bxh, byh, bzh, cxh, cyh, czh)

@njit(**jitflags)
def __mass_flux_accumulated(axh, ayh, azh, bxh, byh, bzh, cxh, cyh, czh):

  cxh[:,:,:] = axh[:,:,:] + bxh[:,:,:]
  cyh[:,:,:] = ayh[:,:,:] + byh[:,:,:] 
  czh[:,:,:] = azh[:,:,:] + bzh[:,:,:]

from numba import njit
from fvms.build_config import jitflags


@njit(**jitflags)
def _fpos(f):
    return max(0.0, f)

@njit(**jitflags)
def _fneg(f):
    return min(0.0, f)


def copy_field(cfg, a, b): 
  __copy_fields(a, b)

@njit(**jitflags)
def __copy_fields(a, b):
  b[:,:,:] = a[:,:,:]


def add_field(cfg, a, b, c): 
  __add_fields(a, b, c)

@njit(**jitflags)
def __add_fields(a, b, c):
  c[:,:,:] = a[:,:,:] + b[:,:,:]
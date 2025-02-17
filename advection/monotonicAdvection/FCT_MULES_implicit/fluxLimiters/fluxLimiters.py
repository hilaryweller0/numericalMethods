# Flux limiters for 1D flux-form advection schemes on uniform grids

# Functions for the numerical methods assuming a one-dimensional, uniform, periodic grid
# A periodic domain is implimented using numpy roll.
# Interface j+1/2 is indexed j so that cell j is between interfaces indexed j-1 and j
# Assumes positive velocity so cell j is upwind of interface j-1
#         |  cell |   
#   j-1   |   j   |   j+1
#        j-1      j

from advectionSchemes import *
import numpy as np
from fluxLimiters.FCT import FCT
from fluxLimiters.MULES import MULES
from fluxLimiters.MULES2 import MULES2

def findMinMax(phid, phi, minPhi, maxPhi):
    """Return phiMin and phiMax for bounded solutions. 
    If minPhi and maxPhi are not none, these are the values
    If phi is not None, find nearest neighbours of phid and phi to determin phiMin and phiMax
    Suitable for c<=1
    If phi is None, just use phid. Suitable for all c but more diffusive."""
    phiMax = maxPhi
    if phiMax is None:
        phiMax = phid
        if phi is not None:
            phiMax = np.maximum(phi, phiMax)
        phiMax = np.maximum(np.roll(phiMax,1), np.maximum(phiMax, np.roll(phiMax,-1)))
    
    phiMin = minPhi
    if phiMin is None:
        phiMin = phid
        if phi is not None:
            phiMin = np.minimum(phi, phiMin)
        phiMin = np.minimum(np.roll(phiMin,1), np.minimum(phiMin, np.roll(phiMin,-1)))

    return phiMin, phiMax


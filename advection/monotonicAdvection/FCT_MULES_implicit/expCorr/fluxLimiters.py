# Flux limiters for 1D flux-form advection schemes on uniform grids

# Functions for the numerical methods assuming a one-dimensional, uniform, periodic grid
# A periodic domain is implimented using numpy roll.
# Interface j+1/2 is indexed j so that cell j is between interfaces indexed j-1 and j
# Assumes positive velocity so cell j is upwind of interface j-1
#         |  cell |   
#   j-1   |   j   |   j+1
#        j-1      j

import numpy as np
import matplotlib.pyplot as plt

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
        phiMax = np.maximum(np.roll(phiMax,1), 
                            np.maximum(phiMax, np.roll(phiMax,-1)))
    
    phiMin = minPhi
    if phiMin is None:
        phiMin = phid
        if phi is not None:
            phiMin = np.minimum(phi, phiMin)
        phiMin = np.minimum(np.roll(phiMin,1), 
                            np.minimum(phiMin, np.roll(phiMin,-1)))

    return phiMin, phiMax

def FCT(A, F1, T, T1, c, 
        options={"minT": None, "maxT": None, "limiterCorr":1, "MULES": False}):
    """Returns the corrected high-order fluxes """
    # Sort out options
    if not isinstance(options, dict):
        options = {}
    minT = options["minT"] if "minT" in options else None
    maxT = options["maxT"] if "maxT" in options else None
    limiterCorr = options["limiterCorr"] if "limiterCorr" in options else 1
    MULES = options["MULES"] if "MULES" in options else False
    
    # The allowable min and max
    if c <= 1:
        TMin, TMax = findMinMax(T1, T, minT, maxT)
    else:
        TMin, TMax = findMinMax(T1, None, minT, maxT)

    # The flux correction to return
    CA = np.zeros_like(A)
    for it in range(limiterCorr):
        # The allowable rise and fall using the bounded solution, Tb
        Tb = T1 - (CA - np.roll(CA,1))
        Qp = TMax - Tb
        Qm = Tb - TMin

        # Sums of influxes and outfluxes
        Pp = np.maximum(0, np.roll(A,1)) - np.minimum(0, A)
        Pm = np.maximum(0, A) - np.minimum(0, np.roll(A,1))
        
        # MULES additions
        Ppp = 0 if ((it == 0) or not MULES) else \
            np.maximum(0, np.roll(CA,1)) - np.minimum(0, CA)
        Pmp = 0 if ((it == 0) or not MULES) else \
            np.maximum(0, CA) - np.minimum(0, np.roll(CA,1))
    
        # Ratios of allowable to HO fluxes
        Rp = np.where(Pp > 1e-12, np.minimum(1, (Qp+Pmp)/np.maximum(Pp,1e-12)), 0)
        Rm = np.where(Pm > 1e-12, np.minimum(1, (Qm+Ppp)/np.maximum(Pm,1e-12)), 0)

        # The flux limiter
        CA += A*np.where(A >= 0, np.minimum(np.roll(Rp,-1), Rm),
                             np.minimum(Rp, np.roll(Rm,-1)))
        A = A - CA
    
    return CA

def MULES(A, F1, T, T1, c, 
        options={"minT": None, "maxT": None, "limiterCorr":1}):
    """Returns the corrected high-order fluxes """
    return FCT(A, F1, T, T1, c, options={**options, "MULES": True})


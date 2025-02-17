import numpy as np
from advectionSchemes import *

def FCT(phi, c, options={"HO":PPMflux, "LO":upwindFlux, "nCorr":1, 
                         "minPhi": None, "maxPhi": None}):
    """Returns the corrected high-order fluxes with nCorr corrections"""
    from fluxLimiters.fluxLimiters import findMinMax
    # Sort out options
    if not isinstance(options, dict):
        options = {}
    HO =  options["HO"] if "HO" in options else PPMflux
    LO =  options["LO"] if "LO" in options else upwindFlux
    nCorr = options["nCorr"] if "nCorr" in options else 1
    minPhi = options["minPhi"] if "minPhi" in options else None
    maxPhi = options["maxPhi"] if "maxPhi" in options else None
    
    # First approximation of the bounded flux and the full HO flux
    fluxB = LO(phi,c, options=options)#{**options, **{'alpha': 1}})
    fluxH = HO(phi,c, options=options)

    # The first bounded solution
    phid = advect(phi, c, fluxB)

    # The allowable min and max
    if c <= 1:
        phiMin, phiMax = findMinMax(phid, phi, minPhi, maxPhi)
    else:
        phiMin, phiMax = findMinMax(phid, None, minPhi, maxPhi)

    # Add a corrected HO flux
    for it in range(nCorr):
        # The antidiffusive fluxes
        A = fluxH - fluxB

        # Sums of influxes ad outfluxes
        Pp = c*(np.maximum(0, np.roll(A,1)) - np.minimum(0, A))
        Pm = c*(np.maximum(0, A) - np.minimum(0, np.roll(A,1)))
        
        # The allowable rise and fall using an updated bounded solution
        if it > 0:
            phid = advect(phi, c, fluxB)
        Qp = phiMax - phid
        Qm = phid - phiMin

        # Ratios of allowable to HO fluxes
        Rp = np.where(Pp > 1e-12, np.minimum(1, Qp/np.maximum(Pp,1e-12)), 0)
        Rm = np.where(Pm > 1e-12, np.minimum(1, Qm/np.maximum(Pm,1e-12)), 0)

        # The flux limiter
        C = np.where(A >= 0, np.minimum(np.roll(Rp,-1), Rm),
                             np.minimum(Rp, np.roll(Rm,-1)))
        fluxB = fluxB + C*A
        
    return fluxB


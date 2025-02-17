import numpy as np
from advectionSchemes import *

def MULES2(phi, c, options={"HO":PPMflux, "LO":upwindFlux, "nCorr":2, 
                         "minPhi": None, "maxPhi": None, "safeStart": False}):
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
    safeStart = options["safeStart"] if "safeStart" in options else False

    # First approximation of the bounded flux and the full HO flux
    fluxB = LO(phi,c, options=options)#{**options, **{'alpha': 1}})
    fluxH = HO(phi,c, options=options)

    # The bounded solution
    phid = advect(phi, c, fluxB)

    # The allowable min and max
    if c <= 1:
        phiMin, phiMax = findMinMax(phid, phi, minPhi, maxPhi)
    else:
        phiMin, phiMax = findMinMax(phid, None, minPhi, maxPhi)

    # The allowable rise and fall
    Qp = phiMax - phid
    Qm = phid - phiMin

    # The antidiffusive fluxes
    A = fluxH - fluxB

    # Iterations of MULES, starting with limiter of 1
    C = np.zeros_like(A) if safeStart else np.ones_like(A)
    for ic in range(nCorr):
        # Ratios of allowable to HO fluxes (differs from FCT)
        CA = C*A
        # Sums of influxes ad outfluxes
        Pp = c*(np.maximum(0, np.roll(A-CA,1)) - np.minimum(0, A-CA))
        Pm = c*(np.maximum(0, A-CA) - np.minimum(0, np.roll(A-CA,1)))

        #Ppp = c*(np.maximum(0, np.roll(CA,1)) - np.minimum(0, CA))
        #Pmp = c*(np.maximum(0, CA) - np.minimum(0, np.roll(CA,1)))
        Ppp = 0 if ((ic == 0) and safeStart) else \
            c*(np.maximum(0, np.roll(CA,1)) - np.minimum(0, CA))
        Pmp = 0 if ((ic == 0) and safeStart) else \
            c*(np.maximum(0, CA) - np.minimum(0, np.roll(CA,1)))

        Rp = np.where(Pp>1e-12, np.minimum(1, (Qp+Pmp)/np.maximum(Pp,1e-12)),0)
        Rm = np.where(Pm>1e-12, np.minimum(1, (Qm+Ppp)/np.maximum(Pm,1e-12)),0)

        # Correction to the flux limiter
        #C = np.minimum(C, np.where(A >= 0, np.minimum(np.roll(Rp,-1), Rm),
        #                          np.minimum(Rp, np.roll(Rm,-1))))
        C = np.where(A >= 0, np.minimum(np.roll(Rp,-1), Rm),
                             np.minimum(Rp, np.roll(Rm,-1)))
    return fluxB + C*A


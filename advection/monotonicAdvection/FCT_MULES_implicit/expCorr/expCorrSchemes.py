# 1D flux-form advection schemes on uniform grids

# Numerical methods using a one-dimensional, uniform, periodic grid
# A periodic domain is implimented using numpy roll.
# HO corrections are either explicit corrections or explicit sources.
# Interface j+1/2 is indexed j so that cell j is between interfaces indexed j-1 and j
# Assumes positive velocity so cell j is upwind of interface j-1
#         |  cell |   
#   j-1   |   j   |   j+1
#        j-1      j

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import numpy as np
from fluxLimiters import *

def advectImplicitExpCorr(T, c, flux, nCorr, split=True, options=None):
    """Advect cell values T using Courant number c using flux for one time step.
    "flux" is callable and is called with arguments T and options=options"""
    # Set the off-centering, aL/aH for the low/high-order solutions
    aL = options["alphaL"] if "alphaL" in options else \
         options["alpha"] if "alpha" in options else alpha
    if (callable(aL)):
        aL = aL(c)
    aH = options["alphaH"] if "alphaH" in options else \
         options["alpha"] if "alpha" in options else 0.5
    limiter = options["limiter"] if "limiter" in options else None
    if (callable(aH)):
        aH = aH(c)

    # The upwind matrix
    M = upwindMatrix(aL*c, len(T))
    
    # The low-order bounded solution, T1, and flux, F1
    F1 = upwindFlux(T,(1-aL)*c)
    T1 = spsolve(M, T - (F1 - np.roll(F1,1)))
    F1 += upwindFlux(T1,aL*c)
    
    # The final version of T to return
    Tnew = T1.copy()
    
    if split:
        # The old time level flux correction as a correction
        F0 = flux(T, (1-aH)*c, options=options) - F1

        # Iterate, updating the HO contribution
        for icorr in range(nCorr):
            F = F0 + flux(Tnew, aH*c, options=options)
            # Under-relax the flux
            if icorr == 0:
                Fprev = F
            else:
                F = 0.5*(F + Fprev)
            # Limit the flux correction if required
            if (callable(limiter)):
                F = limiter(F, F1, T, T1, c, options=options)
            
            # Update T
            Tnew = T1 - (F - np.roll(F,1))
    
    else: # exp Source to the implicit solution
        # First the old time level update
        F = flux(T, (1-aH)*c)
        T = T - (F - np.roll(F,1))
        
        # Iterate, updating the HO contribution
        for icorr in range(nCorr):
            # Calculate the HO correction
            F = flux(Tnew, aH*c, options=options) - upwindFlux(Tnew, aL*c)
            # Solve the full system
            Tnew = spsolve(M, T - (F - np.roll(F,1)))
    
    return Tnew

def upwindMatrix(ac, nx):
    # Matrix for implicit solution of
    # phi_j^{n+1} = phi_j^n - (1-a)*c(phi_j^n - phi_{j-1}^n)
    #                       - a*c*(phi_j^{n+1} - phi_{j-1}^{n+1}
    return diags([-ac*np.ones(nx-1),  # The diagonal for j-1
                 (1+ac)*np.ones(nx), # The diagonal for j
                 [-ac]], # The top right corner for j-1
                 [-1,0,nx-1], # the locations of each of the diagonals
                 shape=(nx,nx), format = 'csr')

def alpha(c):
    "Off-centering for implicit solution"
    return 1-1/np.maximum(c, 1)


def upwindFlux(T, c, options=None):
    return c*T

def CDFlux(T, c, options=None):
    """Returns the second-order centered fluxes for cell values T"""
    return 0.5*c*(T + np.roll(T,-1))

def quasiCubicFlux(T, c, options=None):
    """Returns the quasi-cubic fluxes for cell values T"""
    return c*(2*np.roll(T,-1) + 5*T - np.roll(T,1))/6

def linearUpwindFlux(T, c, options=None):
    """Returns the linear upwind fluxes for cell values T"""
    return c*(np.roll(T,-1) + 4*T - np.roll(T,1))/4


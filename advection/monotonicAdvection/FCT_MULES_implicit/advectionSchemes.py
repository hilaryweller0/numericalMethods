# 1D flux-form advection schemes on uniform grids

# Functions for the numerical methods assuming a one-dimensional, uniform, periodic grid
# A periodic domain is implimented using numpy roll.
# Interface j+1/2 is indexed j so that cell j is between interfaces indexed j-1 and j
# Assumes positive velocity so cell j is upwind of interface j-1
#         |  cell |   
#   j-1   |   j   |   j+1
#        j-1      j

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import numpy as np
from JacobiSolve import *

def advect(phi, c, flux, options=None):
    """Advect cell values phi using Courant number c using flux for one time step.
    If "flux" is callable, it is called with arguments phi, c and options=options"""
    F = flux
    if callable(flux):
        F = flux(phi, c, options=options)
    return phi - c*(F - np.roll(F,1))
    
def PPMflux(phi, c,options=None):
    """Returns the PPM fluxes for cell values phi for Courant number c.
    Face j is at j+1/2 between cells j and j+1"""
    # Integer and remainder parts of the Courant number
    cI = int(c)
    cR = c - cI
    # phi interpolated onto faces
    phiI = 1/12*(-np.roll(phi,1) + 7*phi + 7*np.roll(phi,-1) - np.roll(phi,-2))
    # Move face interpolants to the departure faces
    if cI > 0:
        phiI = np.roll(phiI,cI)
    # Interface fluxes
    F = np.zeros_like(phi)
    # Contribution to the fluxes from full cells between the face and the departure point
    nx = len(F)
    for j in range(nx):
        for i in range(j-cI+1,j+1):
            F[j] += phi[i%nx]/c
    # Ratio of remnamt to total Courant number
    cS = 1
    if c > 1:
        cS = cR/c
    # Contribution from the departure cell
    F += cS*( (1 - 2*cR + cR**2)*phiI \
             + (3*cR - 2*cR**2)*np.roll(phi,cI) \
             + (-cR + cR**2)*np.roll(phiI,1))
    return F

def upwindMatrix(c, a, nx):
    # Matrix for implicit solution of
    # phi_j^{n+1} = phi_j^n - (1-a)*c(phi_j^n - phi_{j-1}^n)
    #                       - a*c*(phi_j^{n+1} - phi_{j-1}^{n+1}
    return diags([-a*c*np.ones(nx-1),  # The diagonal for j-1
                 (1+a*c)*np.ones(nx), # The diagonal for j
                 [-a*c]], # The top right corner for j-1
                 [-1,0,nx-1], # the locations of each of the diagonals
                 shape=(nx,nx), format = 'csr')

def alpha(c):
    "Off-centering for implicit solution"
    return 1-1/np.maximum(c, 1)

def upwindFlux(phi, c, options=None):
    """Returns the first-order upwind fluxes for cell values phi and Courant number c
    Implicit or explicit depending on Courant number"""
    if not isinstance(options, dict):
        options = {}
    explicit =  options["explicit"] if "explicit" in options else (c <= 1)
    if explicit:
        return phi
    nx = len(phi)
    # Off centering for Implicit-Explicit
    a = options["alpha"] if "alpha" in options else alpha
    if callable(a):
        a = a(c)
    M = upwindMatrix(c, a, nx)
    # Solve the implicit problem
    phiNew = phi.copy()
    Jacobi = options["Jacobi"] if "Jacobi" in options else False
    if Jacobi:
        JacobiSolve(M, phi - (1-a)*c*(phi - np.roll(phi,1)), phiNew, maxIt=10)
    else:
        phiNew = spsolve(M, phi - (1-a)*c*(phi - np.roll(phi,1)))
    # Back-substitute to get the implicit fluxes
    return (1-a)*phi + a*phiNew

def CDFlux(phi, c, options=None):
    """Returns the second-order centered fluxes for cell values phi and Courant number c
    Implicit or explicit depending on Courant number"""
    if not isinstance(options, dict):
        options = {}
    explicit =  options["explicit"] if "explicit" in options else (c <= 1)
    if explicit:
        return 0.5*(phi + np.roll(phi,-1))
    nx = len(phi)
    # Off centering for 2nd-order Implicit-Explicit
    a = options["alpha"] if "alpha" in options else 0.5
    # Matrix for implicit solution of
    # phi_j^{n+1} = phi_j^n - (1-a)*c/2*(phi_{j+1}^n - phi_{j-1}^n)
    #                       - a*c/2*(phi_{j+1}^{n+1} - phi_{j-1}^{n+1}
    M = diags([a*c/2, # the bottom left corder for j+1
               -a*c/2*np.ones(nx-1),  # The diagonal for j-1
               np.ones(nx), # The diagonal for j
               a*c/2*np.ones(nx-1), # The diagonal for j+1
               -a*c/2], # The top right corner for j-1
              [-nx+1, -1,0,1,nx-1], # the locations of each of the diagonals
               shape=(nx,nx), format = 'csr')
    # Solve the implicit problem
    phiNew = spsolve(M, phi - (1-a)*c/2*(np.roll(phi,-1) - np.roll(phi,1)))
    # Back-substitute to get the implicit fluxes
    return (1-a)/2*(phi + np.roll(phi,-1)) + a/2*(phiNew + np.roll(phiNew,-1))

def quasiCubicFlux(phi, c, options=None):
    """Returns the quasi-cubic fluxes for cell values phi and Courant number c
    Implicit or explicit depending on Courant number or options"""
    qcFlux = lambda T: (2*np.roll(T,-1) + 5*T - np.roll(T,1))/6
    if not isinstance(options, dict):
        options = {}
    explicit =  options["explicit"] if "explicit" in options else (c <= 1)
    if explicit:
        return qcFlux(phi)
    nx = len(phi)
    # Off centering for 2nd-order Implicit-Explicit
    a = options["alpha"] if "alpha" in options else 0.5
    # Matrix for implicit solution of
    # phi_j^{n+1} = phi_j^n - (1-a)*c*(phi_{j+1/2}^n - phi_{j-1/2}^n)
    #                       - a*c*(phi_{j+1/2}^{n+1} - phi_{j-1/2}^{n+1}
    #             = phi_j^n - (1-a)*c/6*(2phi_{j+1} + 3phi_j - 6phi_{j-1} + phi{j-2})^n
    #                       - a*c/6*(2phi_{j+1} + 3phi_j - 6phi_{j-1} + phi{j-2})^{n+1}
    M = diags([a*c/3, # the bottom left corder for j+1
               a*c/6*np.ones(nx-2),  # The diagonal for j-2
               -a*c*np.ones(nx-1),  # The diagonal for j-1
               (1 + a*c/2)*np.ones(nx), # The diagonal for j
               a*c/3*np.ones(nx-1), # The diagonal for j+1
               [a*c/6, a*c/6],# Top right for j-2
               -a*c], # The top right corner for j-1
              [-nx+1, -2,-1,0,1,nx-2,nx-1], # the locations of each of the diagonals
               shape=(nx,nx), format = 'csr')
    #M = upwindMatrix(c, a, nx)
    # Solve the implicit problem
    phiNew = spsolve(M, phi - (1-a)*c*(qcFlux(phi) - np.roll(qcFlux(phi),1)))
    # Back-substitute to get the implicit fluxes
    return (1-a)*qcFlux(phi) + a*qcFlux(phiNew)

def linearUpwindFlux(phi, c, options=None):
    """Returns the linear upwind fluxes for cell values phi and Courant number c
    Implicit or explicit depending on Courant number or options"""
    luFlux = lambda T: (np.roll(T,-1) + 4*T - np.roll(T,1))/4
    if not isinstance(options, dict):
        options = {}
    explicit =  options["explicit"] if "explicit" in options else (c <= 1)
    if explicit:
        return qcFlux(phi)
    nx = len(phi)
    # Off centering for 2nd-order Implicit-Explicit
    a = options["alpha"] if "alpha" in options else 0.5
    # Matrix for implicit solution of
    # phi_j^{n+1} = phi_j^n - (1-a)*c*(phi_{j+1/2}^n - phi_{j-1/2}^n)
    #                       - a*c*(phi_{j+1/2}^{n+1} - phi_{j-1/2}^{n+1}
    #             = phi_j^n - (1-a)*c/4*(phi_{j+1} + 3phi_j - 5phi_{j-1} + phi{j-2})^n
    #                       - a*c/4*(phi_{j+1} + 3phi_j - 5phi_{j-1} + phi{j-2})^{n+1}
    M = diags([a*c/4, # the bottom left corder for j+1
               a*c/4*np.ones(nx-2),  # The diagonal for j-2
               -5/4*a*c*np.ones(nx-1),  # The diagonal for j-1
               (1 + 3/4*a*c)*np.ones(nx), # The diagonal for j
               a*c/4*np.ones(nx-1), # The diagonal for j+1
               [a*c/4, a*c/4],# Top right for j-2
               -5/4*a*c], # The top right corner for j-1
              [-nx+1, -2,-1,0,1,nx-2,nx-1], # the locations of each of the diagonals
               shape=(nx,nx), format = 'csr')
    #M = upwindMatrix(c, a, nx)
    # Solve the implicit problem
    phiNew = spsolve(M, phi - (1-a)*c*(luFlux(phi) - np.roll(luFlux(phi),1)))
    # Back-substitute to get the implicit fluxes
    return (1-a)*luFlux(phi) + a*luFlux(phiNew)

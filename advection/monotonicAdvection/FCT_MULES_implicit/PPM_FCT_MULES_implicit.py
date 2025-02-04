#!/usr/bin/env python
# coding: utf-8

# # PPM with implicit FCT and MULES
# Apply FCT and MULES to PPM with a large time step. 
# The first application of FCT should use an implicit upwind method for the bounded solution. FCT then creates a bounded correction of PPM. This can be used as the bounded solution to apply FCT again to the PPM solution. Will this process converge to a more accurate bounded solution?

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.rcParams['figure.dpi'] = 300


# In[2]:


# Functions for the numerical methods assuming a one-dimensional, uniform, periodic grid
# A periodic domain is implimented using numpy roll.
# Interface j+1/2 is indexed j so that cell j is between interfaces indexed j-1 and j
# Assumes positive velocity so cell j is upwind of interface j-1
#         |  cell |   
#   j-1   |   j   |   j+1
#        j-1      j

def advect(phi, c, flux, options=None):
    """Advect cell values phi using Courant number c using flux"""
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
        #F[j] += sum(phi[j-cI+1:j+1])
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
    a = alpha(c)
    M = upwindMatrix(c, a, nx)
    # Solve the implicit problem
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

def FCT(phi, c, options={"HO":PPMflux, "LO":upwindFlux, "nCorr":1, 
                         "minPhi": None, "maxPhi": None}):
    """Returns the corrected high-order fluxes with nCorr corrections"""
    # Sort out options
    if not isinstance(options, dict):
        options = {}
    HO =  options["HO"] if "HO" in options else PPMflux
    LO =  options["LO"] if "LO" in options else upwindFlux
    nCorr = options["nCorr"] if "nCorr" in options else 1
    minPhi = options["minPhi"] if "minPhi" in options else None
    maxPhi = options["maxPhi"] if "maxPhi" in options else None
    
    # First approximation of the bounded flux and the full HO flux
    fluxB = LO(phi,c, options=options)
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

def MULES(phi, c, options={"HO":PPMflux, "LO":upwindFlux, "nCorr":2, 
                         "minPhi": None, "maxPhi": None}):
    """Returns the corrected high-order fluxes with nCorr corrections"""
    # Sort out options
    if not isinstance(options, dict):
        options = {}
    HO =  options["HO"] if "HO" in options else PPMflux
    LO =  options["LO"] if "LO" in options else upwindFlux
    nCorr = options["nCorr"] if "nCorr" in options else 1
    minPhi = options["minPhi"] if "minPhi" in options else None
    maxPhi = options["maxPhi"] if "maxPhi" in options else None
    # First approximation of the bounded flux and the full HO flux
    fluxB = LO(phi,c, options=options)
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

    # Sums of influxes ad outfluxes
    Pp = c*(np.maximum(0, np.roll(A,1)) - np.minimum(0, A))
    Pm = c*(np.maximum(0, A) - np.minimum(0, np.roll(A,1)))

    # Iterations of MULES, starting with limiter of 1
    C = np.ones_like(A)
    for ic in range(nCorr):
        # Ratios of allowable to HO fluxes (definintion differs from FCT, using CA, not A)
        CA = C*A
        Ppp = c*(np.maximum(0, np.roll(CA,1)) - np.minimum(0, CA))
        Pmp = c*(np.maximum(0, CA) - np.minimum(0, np.roll(CA,1)))
        #Ppp = 0 if (ic == 0) else c*(np.maximum(0, np.roll(CA,1)) - np.minimum(0, CA))
        #Pmp = 0 if (ic == 0) else c*(np.maximum(0, CA) - np.minimum(0, np.roll(CA,1)))

        Rp = np.where(Pp > 1e-12, np.minimum(1, (Qp+Pmp)/np.maximum(Pp,1e-12)), 0)
        Rm = np.where(Pm > 1e-12, np.minimum(1, (Qm+Ppp)/np.maximum(Pm,1e-12)), 0)

        # Correction to the flux limiter
        C = np.minimum(C, np.where(A >= 0, np.minimum(np.roll(Rp,-1), Rm),
                                  np.minimum(Rp, np.roll(Rm,-1))))
    return fluxB + C*A

def MULESimplicit(phi, c, options={"nCorr":2, "minPhi": None, "maxPhi": None}):
    """Returns the Implicit MULES corrected central-difference fluxes with nCorr corrections"""
    # Sort out options
    if not isinstance(options, dict):
        options = {}
    nCorr = options["nCorr"] if "nCorr" in options else 1
    minPhi = options["minPhi"] if "minPhi" in options else None
    maxPhi = options["maxPhi"] if "maxPhi" in options else None
    
    # First approximation of the bounded flux
    fluxB = upwindFlux(phi,c)

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

    # The matrix for the implicit part
    a = alpha(c)
    M = upwindMatrix(c, a, len(phi))
    
    # First approximation of the new solution
    phiNew = phi.copy()
    
    # The high order and antidiffusive fluxes
    fluxH = (1-a)*0.5*(phi + np.roll(phi,-1))\
          + a*0.5*(phiNew + np.roll(phiNew,-1))
    A = fluxH - fluxB

    # Iterations of MULES, starting with limiter of 1
    C = np.ones_like(A)
    for ic in range(nCorr):      
        # Sums of unlimited influxes ad outfluxes
        Pp = c*(np.maximum(0, np.roll(A,1)) - np.minimum(0, A))
        Pm = c*(np.maximum(0, A) - np.minimum(0, np.roll(A,1)))
    
        # Ratios of allowable to HO fluxes (definintion differs from FCT, using CA, not A)
        CA = C*A
        Ppp = 0 if (ic == 0) else c*(np.maximum(0, np.roll(CA,1)) - np.minimum(0, CA))
        Pmp = 0 if (ic == 0) else c*(np.maximum(0, CA) - np.minimum(0, np.roll(CA,1)))
        #Ppp = c*(np.maximum(0, np.roll(CA,1)) - np.minimum(0, CA))
        #Pmp = c*(np.maximum(0, CA) - np.minimum(0, np.roll(CA,1)))

        Rp = np.where(Pp > 1e-12, np.minimum(1, (Qp+Pmp)/np.maximum(Pp,1e-12)), 0)
        Rm = np.where(Pm > 1e-12, np.minimum(1, (Qm+Ppp)/np.maximum(Pm,1e-12)), 0)

        # Correction to the flux limiter
        C = np.minimum(C, np.where(A >= 0, np.minimum(np.roll(Rp,-1), Rm),
                                  np.minimum(Rp, np.roll(Rm,-1))))

        # Implicit solution
        CA = C*A
        phiNew = spsolve(M, phi - (1-a)*c*(phi - np.roll(phi,1)) - c*(CA - np.roll(CA,1)))

        # The new antidiffusive fluxes
        fluxH = (1-a)*0.5*(phi + np.roll(phi,-1))\
              + a*0.5*(phiNew + np.roll(phiNew,-1))
        A = fluxH - fluxB
    return fluxB + C*A


# In[3]:


# Initial condition functions

def combi(x):
    "Initial conditions consisting of a square wave and a bell"
    return np.where((x > 0) & (x < 0.5), 0.5*(1-np.cos(2*np.pi*x/.5)),
                    np.where((x > 0.6) & (x < 0.8), 1., 0.))

def square(x):
    "Initial conditions consisting of a square wave"
    return np.where((x > 0) & (x < 0.4), 1., 0.)

def cosBell(x):
    "Initial conditions consisting of a  bell"
    return np.where((x > 0) & (x < 0.4), 0.5*(1-np.cos(2*np.pi*x/.4)), 0)


# In[4]:


# Solutions

def compareSchemes(params, fluxes, options, labels):
    """Solve the advection equation for various schemes and plot the results
    params: dict with entries "nt", "nx", "c", "initialConditions", "title", "fileName"
    fluxes: List of flux functions
    options: List of dictionaries to send to the flux functions
    labels:  Labels for the legend of the graph for each flux function
    """
    dt = params["c"]/params["nx"]
    dx = 1/params["nx"]
    print('Solving the advection equation for', dt*params["nt"],
          'revolutions of a periodic domain with spatial resolution', dx)
    x = np.arange(0,1, dx)
    phi0 = params["initialConditions"](x)
    phiE = params["initialConditions"]((x-dt*params["nt"])%1)
    #fig,ax = plt.subplots(1,2, figsize=(12,4), layout='constrained')
    #fig.sup
    plt.title(params["title"]+'\nc = '+str(round(params["c"],2))+' nx = '
              +str(params["nx"]) + ' nt = '+str(params["nt"]))
    plt.plot(x, phi0, 'k--', label = 't=0')
    plt.plot(x, phiE, 'k', label='t='+str(round(dt*params["nt"],2)))
    plt.axhline(y=0, color='k', ls=':', lw=0.5)
    plt.axhline(y=1, color='k', ls=':', lw=0.5)

    # Run all the schemes and plot the results
    lines = ['k-o', 'r-+', 'b-x', 'g--s', 'c--', 'm:', 'k:', 'r:', 'b:']
    for flux, name, option, line in zip(fluxes, labels, options, lines):
        phi = phi0.copy()
        for it in range(params["nt"]):
            phi = advect(phi, params["c"], flux, options=option)
        plt.plot(x, phi, line, label=name)
        #ax[1].plot(x, phi - phiE, label=name)
    
    plt.legend()#bbox_to_anchor=(1.1, 1))
    #ax[0].set(ylabel=r'$\psi$', title = 'Totals', xlim=[0,1])
    #ax[1].set(ylabel='Error', title = 'Errors', xlim=[0,1])
    plt.xlim([0,1])
    plt.savefig(params["fileName"])
    plt.show()

# calculate the number of time steps from the number of revolutions, nx and c
nt = lambda nRevs, nx, c : int(nRevs*nx/c)

# Specific plots
# Explicit PPM with FCT and  MULES
compareSchemes(
    {"nt": nt(1,40,.4), "nx":40, "c":0.4, "initialConditions":combi,
     "title": "Explicit PPM Advection with FCT/MULES",
     "fileName": "plots/PPM_c04_FCT_MULES.pdf"},
    [FCT, FCT, MULES, MULES, PPMflux],
    [{"nCorr": 1}, {"nCorr": 2}, {"nCorr": 1}, {"nCorr": 2}, {}],
    ['with 1 FCT', 'with 2 FCT', 'with 1 explicit MULES', 'with 2 explicit MULES', 'PPM'])

# Explicit Centred differences with FCT and  MULES (not explicit)
compareSchemes(
    {"nt": nt(1,40,.4), "nx":40, "c":0.4, "initialConditions":combi,
     "title": r'Centred differences with FCT/MULES with $\alpha=\frac{1}{2}$ for CD',
     "fileName": "plots/CD_c04_FCT_MULES.pdf"},
    [FCT, FCT, MULES, MULES, CDFlux],
    [{'HO': CDFlux, "nCorr": 1, "explicit": False, "alpha": 0.5},
     {'HO': CDFlux, "nCorr": 2, "explicit": False, "alpha": 0.5},
     {'HO': CDFlux, "nCorr": 1, "explicit": False, "alpha": 0.5},
     {'HO': CDFlux, "nCorr": 2, "explicit": False, "alpha": 0.5},
     {"explicit": False, "alpha": 0.5}],
    ['CD with 1 FCT', 'CD with 2 FCT', 'CD with 1 MULES', 'CD with 2 MULES', 
     r'CD with $\alpha=\frac{1}{2}$'])

# Explicit quasi-cubic with FCT and MULES (not explicit)
compareSchemes(
    {"nt" : nt(1,40,.4), "nx": 40, "c": 0.4, "initialConditions": combi,
     "title": r'Quasi-cubic with FCT/MULES with $\alpha=\frac{1}{2}$ for qC',
     "fileName": "plots/qC_c04_FCT_MULES.pdf"},
    [FCT, FCT, MULES, MULES, quasiCubicFlux],
    [{"nCorr": 1, "HO": quasiCubicFlux, "explicit": False, "alpha": 0.5},
     {"nCorr": 2, "HO": quasiCubicFlux, "explicit": False, "alpha": 0.5},
     {"nCorr": 1, "HO": quasiCubicFlux, "explicit": False, "alpha": 0.5},
     {"nCorr": 2, "HO": quasiCubicFlux, "explicit": False, "alpha": 0.5},
     {"explicit": False, "alpha": 0.5}],
    ['with 1 FCT', 'with 2 FCT', 'with 1 MULES', 
     'with 2 MULES', r'cubic, $\alpha=0.5$'])

# Explicit linear upwind with FCT and MULES (not explicit)
compareSchemes(
    {"nt" : nt(1,40,.4), "nx": 40, "c": 0.4, "initialConditions": combi,
     "title": r'Linear-upwind with FCT/MULES with $\alpha=\frac{1}{2}$ for qC',
     "fileName": "plots/lu_c04_FCT_MULES.pdf"},
    [FCT, FCT, MULES, MULES, linearUpwindFlux],
    [{"nCorr": 1, "HO": linearUpwindFlux, "explicit": False, "alpha": 0.5},
     {"nCorr": 2, "HO": linearUpwindFlux, "explicit": False, "alpha": 0.5},
     {"nCorr": 1, "HO": linearUpwindFlux, "explicit": False, "alpha": 0.5},
     {"nCorr": 2, "HO": linearUpwindFlux, "explicit": False, "alpha": 0.5},
     {"explicit": False, "alpha": 0.5}],
    ['with 1 FCT', 'with 2 FCT', 'with 1 MULES', 
     'with 2 MULES', r'linear uwpind, $\alpha=0.5$'])

# PPM with iterations of implicit, monotonic FCT with c = 1.4
compareSchemes(
    {"nt": 28, "nx":40, "c":40/28, "initialConditions":combi,
     "title": "Implicit Advection with FCT",
     "fileName": "plots/PPM_c14_FCT.pdf"},
    [PPMflux, upwindFlux, FCT, FCT, FCT],
    [{}, {}, {"nCorr": 1}, {"nCorr": 2}, {"nCorr": 3}],
    ['PPM', 'upwind', 'PPM with 1 FCT', 'PPM with 2 FCT', 'PPM with 3 FCT'])

# PPM with iterations of implicit, monotonic MULES with c = 1.4
compareSchemes(
    {"nt": 28, "nx":40, "c":40/28, "initialConditions":combi,
     "title": "Implicit Advection with MULES",
     "fileName": "plots/PPM_c14_MULES.pdf"},
    [PPMflux, upwindFlux, MULES, MULES, MULES],
    [{}, {}, {"nCorr": 1}, {"nCorr": 2}, {"nCorr": 3}],
    ['PPM', 'upwind', 'PPM with 1 MULES', 'PPM with 2 MULES', 'PPM with 3 MULES'])

# PPM with iterations of implicit, monotonic FCT with c = 2.4
compareSchemes(
    {"nt": 32, "nx":80, "c":80/32, "initialConditions":combi,
     "title": "Implicit Advection with FCT",
     "fileName": "plots/PPM_c24_FCT.pdf"},
    [PPMflux, upwindFlux, FCT, FCT, FCT, FCT, FCT, FCT],
    [{}, {}, {"nCorr": 1}, {"nCorr": 2}, {"nCorr": 3}, {"nCorr": 4}, {"nCorr": 5},
     {"nCorr": 6}],
    ['PPM', 'upwind', 'PPM with 1 FCT', 'PPM with 2 FCT', 'PPM with 3 FCT',
     'PPM with 4 FCT', 'PPM with 5 FCT', 'PPM with 6 FCT'])

# PPM with iterations of implicit, monotonic MULES with c = 2.4
compareSchemes(
    {"nt": 32, "nx":80, "c":80/32, "initialConditions":combi,
     "title": "Implicit Advection with MULES",
     "fileName": "plots/PPM_c24_MULES.pdf"},
    [PPMflux, upwindFlux, MULES, MULES, MULES, MULES, MULES, MULES],
    [{}, {}, {"nCorr": 1}, {"nCorr": 2}, {"nCorr": 3}, {"nCorr": 4}, {"nCorr": 5},
     {"nCorr": 6}],
    ['PPM', 'upwind', 'PPM with 1 MULES', 'PPM with 2 MULES', 'PPM with 3 MULES',
     'PPM with 4 MULES', 'PPM with 5 MULES', 'PPM with 6 MULES'])

# Comparison with Amber's results on doubleFCT.py
compareSchemes(
    {"nt" : int(100/6.25), "nx": 40, "c": 2.5, "initialConditions": combi,
     "title": "Quasi-cubic advection with Trapezoidal-implicit",
     "fileName": "plots/qC_c2p5_FCT.pdf"},
    [quasiCubicFlux, FCT, FCT, FCT, FCT, FCT, FCT],
    [{"explicit": False, "alpha": 0.5},
     {"nCorr": 1, "HO": quasiCubicFlux, "explicit": False, "alpha": 0.5},
     {"nCorr": 2, "HO": quasiCubicFlux, "explicit": False, "alpha": 0.5},
     {"nCorr": 3, "HO": quasiCubicFlux, "explicit": False, "alpha": 0.5},
     {"nCorr": 4, "HO": quasiCubicFlux, "explicit": False, "alpha": 0.5},
     {"nCorr": 5, "HO": quasiCubicFlux, "explicit": False, "alpha": 0.5},
     {"nCorr": 6, "HO": quasiCubicFlux, "explicit": False, "alpha": 0.5}],
    [r'cubic, $\alpha=0.5$', 'with 1 FCT', 'with 2 FCT', 'with 3 FCT', 
     'with 4 FCT', 'with 5 FCT', 'with 6 FCT'])


# In[5]:


# Debugging code for Amber
c = 2.5
nx = 80
x = np.arange(0,1, 1/nx)
phi = combi(x)
phi = advect(phi, c, PPMflux)
print(phi)


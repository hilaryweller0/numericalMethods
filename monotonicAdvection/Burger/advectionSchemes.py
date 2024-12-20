# Numerical schemes for simulating advection for outer code advection.py

from __future__ import absolute_import, division, print_function

import numpy as np

def semiLagrangian(phiOld, c, d, nt):
    "Advection of profile in phiOld using semi-Lagrangian with cubic-Lagrange"
    "interpolating using Courant number c"
    "for nt time steps with periodic boundary conditions"
    "and non-dimensional diffusion coefficient d"
    "Hilary Weller"
    
    nx = len(phiOld)
    dx = 1/nx
    x = np.arange(0.,1.,dx)
    dt = c*dx

    # new time-step array for phi
    phi = phiOld.copy()
    
    # semi-Lagrangian for nt time steps
    for it in range(nt):
        # First apply diffusion
        phiOld = phiOld + d*(np.roll(phiOld,1) - 2*phiOld + np.roll(phiOld,-1))
        for i in range(nx):
            # Find the velocity half way back along the trajectory. First find
            # the trajectory mid point assuming the velocity of the arrival point
            xdhalf = x[i] - 0.5*phi[i]*dt
            # grid point of the mid trajectory point
            k = int(np.floor(xdhalf/dx))
            # Interpolation parameter
            beta = i-k-phi[i]*dt/dx
            # Interpolate onto trajectory mid point poing with cubic-Lagrange
            uhalf = -1/6*beta*(1-beta)*(2-beta)*phiOld[(k-1)%nx] \
                   + .5*(1+beta)*(1-beta)*(2-beta)*phiOld[k%nx]   \
                   + .5*(1+beta)*beta*(2-beta)*phiOld[(k+1)%nx]   \
                   - 1/6*(1+beta)*beta*(1-beta)*phiOld[(k+2)%nx]

            # Find the departure point using the velocity half way along the
            # trajectory
            xd = x[i] - uhalf*dt
            # The grid point of the departure point
            k = int(np.floor(xd/dx))
            # Interpolation parameter
            beta = i-k - uhalf*dt/dx
            # Interpolate onto departure poing with cubic-Lagrange
            phi[i] = -1/6*beta*(1-beta)*(2-beta)*phiOld[(k-1)%nx] \
                   + .5*(1+beta)*(1-beta)*(2-beta)*phiOld[k%nx]   \
                   + .5*(1+beta)*beta*(2-beta)*phiOld[(k+1)%nx]   \
                   - 1/6*(1+beta)*beta*(1-beta)*phiOld[(k+2)%nx]
        # Update phi for next time-step
        phiOld = phi.copy()

    return phi

def MPDATA(phiOld, c, d, nt, explicit=1):
    "Advection of profile in phiOld using MPDATA using Courant number c"
    "for nt time steps with periodic boundary conditions."
    "and diffusion coefficient d"
    "Hilary Weller"
    
    nx = len(phiOld)

    # new time-step arrays for phi
    phi = phiOld.copy()
    
    # Arrays for the fluxes and the ante-diffusive velocities (at j+1/2)
    flux = np.zeros(nx)
    v = np.zeros(nx)
    
    # time steps
    for it in range(nt):
        # First apply diffusion
        phiOld = phiOld + d*(np.roll(phiOld,1) - 2*phiOld + np.roll(phiOld,-1))
        # first-order upwind pass
        if explicit == 1:
            for i in range(nx):
                phi[i] = phiOld[i] \
                       -0.5*c*(phiOld[(i)%nx]**2 - phiOld[(i-1)%nx]**2)
        else:
            phi = BTBS(phiOld, c, 0, 1)
        
        # Ante-diffusive velocities
        for i in range(nx):
            v[i] = (c*phi[i] - (c*phi[i])**2)*(phi[(i+1)%nx] - phi[i])\
                             /(phi[(i+1)%nx] + phi[i] + 1e-10)

        # Limit ante-diffusive velocities
        for i in range(nx):
            v[i] = min(v[i], 1/c)
            v[i] = max(v[i], -1/c)

        # Corrective fluxes
        for i in range(nx):
            flux[i] = 0.5*v[i]*(phi[(i+1)%nx] + phi[i]) \
                    - 0.5*abs(v[i])*(phi[(i+1)%nx] - phi[i])

        # Second-order correction
        for i in range(nx):
            phi[i] -= flux[i] - flux[(i-1)%nx]

        phiOld = phi.copy()

    return phi

def LaxWendroff(phiOld, c, d, nt):
    nx = len(phiOld)

    # new time-step arrays for phi
    phi = phiOld.copy()
    
    # time steps
    for it in range(nt):
        # First apply diffusion
        phiOld = phiOld + d*(np.roll(phiOld,1) - 2*phiOld + np.roll(phiOld,-1))

        for j in range(nx):
            phiPlus = 0.5*(1+c)*phiOld[j] + 0.5*(1-c)*phiOld[(j+1)%nx]
            phiMinus = 0.5*(1+c)*phiOld[(j-1)%nx] + 0.5*(1-c)*phiOld[j]
            phi[j] = phiOld[j] - 0.5*c*(phiPlus**2 - phiMinus**2)

        phiOld = phi.copy()

    return phi

def FTBS(phiOld, c, d, nt):
    nx = len(phiOld)

    # new time-step arrays for phi
    phi = phiOld.copy()
    
    # time steps
    for it in range(nt):
        # First apply diffusion
        phiOld = phiOld + d*(np.roll(phiOld,1) - 2*phiOld + np.roll(phiOld,-1))

        for j in range(nx):
            phi[j] = phiOld[j] - 0.5*c*(phiOld[j]**2 - phiOld[(j-1)%nx]**2)
        phiOld = phi.copy()

    return phi

def BTBS(phi, c, d, nt):
    
    nx = len(phi)
    
    # Solution for nt time steps
    for it in range(nt):
        # matrix for the implicit solution
        M = np.zeros([nx,nx])
        for i in range(nx):
            M[i][i] = 1 + 0.5*c*phi[i]
            M[i][(i-1)%nx] = -0.5*c*phi[(i-1)%nx]

        phi = np.linalg.solve(M, phi)
    
    return phi

def FBTCS(phiOld, c, d, nt):
    nx = len(phiOld)

    # new time-step arrays for phi
    phi = phiOld.copy()
    phiTmp = phiOld.copy()
    
    # time steps
    for it in range(nt):
        # First apply diffusion
        phiOld = phiOld + d*(np.roll(phiOld,1) - 2*phiOld + np.roll(phiOld,-1))
        phiTmp = phiOld.copy()

        for iter in range(2):
            for j in range(nx):
                phiL = 0.5*(phiTmp[(j-1)%nx] + phiTmp[j])
                phiR = 0.5*(phiTmp[(j+1)%nx] + phiTmp[j])
                phi[j] = phiOld[j] - 0.5*c*(phiR**2 - phiL**2)
            phiTmp = phi.copy()

        phiOld = phi.copy()

    return phi

def TVDvanLeer(phiOld, c, d, nt):
    nx = len(phiOld)

    # new time-step arrays for phi
    phi = phiOld.copy()
    
    # time steps
    for it in range(nt):
        # First apply diffusion
        phiOld = phiOld + d*(np.roll(phiOld,1) - 2*phiOld + np.roll(phiOld,-1))

        # Advection for all cells
        for j in range(nx):
            phiPlusHi  = 0.5*(1+c)*phiOld[j] + 0.5*(1-c)*phiOld[(j+1)%nx]
            phiMinusHi = 0.5*(1+c)*phiOld[(j-1)%nx] + 0.5*(1-c)*phiOld[j]
            phiPlusLo = phiOld[j]
            phiMinusLo = phiOld[(j-1)%nx]
            ratioPlus = 0
            ratioMinus = 0
            if phiOld[(j+1)%nx] - phiOld[(j)%nx] != 0:
                ratioPlus = (phiOld[j] - phiOld[(j-1)%nx]) \
                            /(phiOld[(j+1)%nx] - phiOld[(j)%nx])
            if phiOld[(j)%nx] - phiOld[(j-1)%nx] != 0:
                ratioMinus = (phiOld[(j-1)%nx] - phiOld[(j-2)%nx]) \
                            /(phiOld[(j)%nx] - phiOld[(j-1)%nx])
            
            limiterPlus = (ratioPlus + abs(ratioPlus))/(1+abs(ratioPlus))
            limiterMinus = (ratioMinus + abs(ratioMinus))/(1+abs(ratioMinus))
            
            phiPlus = limiterPlus*phiPlusHi + (1-limiterPlus)*phiPlusLo
            phiMinus = limiterMinus*phiMinusHi + (1-limiterMinus)*phiMinusLo
            
            phi[j] = phiOld[j] - 0.5*c*(phiPlus**2 - phiMinus**2)

        phiOld = phi.copy()

    return phi



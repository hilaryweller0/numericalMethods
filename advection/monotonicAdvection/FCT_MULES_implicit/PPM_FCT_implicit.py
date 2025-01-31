#!/usr/bin/env python
# coding: utf-8

# # PPM with implicit FCT
# Apply FCT to PPM with a large time step. The first application of FCT should use an implicit upwind method for the bounded solution. FCT then creates a bounded correction of PPM. This can be used as the bounded solution to apply FCT again to the PPM solution. Will this process converge to a more accurate bounded solution?

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


# In[ ]:


# Functions for the numerical methods assuming a one-dimensional, uniform, periodic grid

def PPMflux(phi, c):
    """Returns the PPM fluxes for cell values phi for Courant number c"""
    # phi interpolated onto faces
    phiI = 1/12*(-np.roll(phi,1) + 7*phi + 7*np.roll(phi,-1) - np.roll(phi,-2))
    return (1 - 2*c + c**2)*phiI \
         + (3*c - 2*c**2)*phi \
         + (-c + c**2)*np.roll(phiI,1)

def upwindFlux(phi, c):
    """Returns the first-order upwind fluxes for cell values phi and Courant number c"""
    return phi

def FCT(phi, c, HO=PPMflux, LO=upwindFlux, nCorr=1):
    """Returns the corrected high-order fluxes with nCorr corrections"""
    # First approximation of the bounded flux and the full HO flux
    fluxB = LO(phi,c)
    fluxH = HO(phi,c)

    # Add a corrected HO flux
    for it in range(nCorr):
        # The bounded solution
        phid = advect(phi, c, fluxB)
    
        # The allowable min and max
        phiMax = np.maximum(phi, phid)
        phiMax = np.maximum(np.roll(phiMax,1), np.maximum(phiMax, np.roll(phiMax,-1)))
        phiMin = np.minimum(phi, phid)
        phiMin = np.minimum(np.roll(phiMin,1), np.minimum(phiMin, np.roll(phiMin,-1)))

        # The antidiffusive fluxes
        A = fluxH - fluxB

        # Sums of influxes ad outfluxes
        Pp = c*np.maximum(0, np.roll(A,1)) - np.minimum(0, A)
        Pm = c*np.maximum(0, A) - np.minimum(0, np.roll(A,1))

        # The allowable rise and fall
        Qp = phiMax - phid
        Qm = phid - phiMin

        # Ratios of allowable to HO fluxes
        Rp = np.where(Pp > 1e-12, np.minimum(1, Qp/np.maximum(Pp,1e-12)), 0)
        Rm = np.where(Pm > 1e-12, np.minimum(1, Qm/np.maximum(Pm,1e-12)), 0)

        # The flux limiter
        C = np.where(A >= 0, np.minimum(np.roll(Rp,-1), Rm), np.minimum(Rp, np.roll(Rm,-1)))
        fluxB = fluxB + C*A
    return fluxB

def advect(phi, c, flux):
    """Advect cell values phi using Courant number c using flux"""
    F = flux
    if callable(flux):
        F = flux(phi, c)
    return phi - c*(F - np.roll(F,1))


# In[ ]:


# Initial condition functions

def combi(x):
    "Initial conditions consisting of a square wave and a bell"
    return np.where((x > 0) & (x < 0.4), 0.5*(1-np.cos(2*np.pi*x/.4)),
                    np.where((x > 0.5) & (x < 0.7), 1., 0.))

def square(x):
    "Initial conditions consisting of a square wave"
    return np.where((x > 0) & (x < 0.4), 1., 0.)

def cosBell(x):
    "Initial conditions consisting of a  bell"
    return np.where((x > 0) & (x < 0.4), 0.5*(1-np.cos(2*np.pi*x/.4)), 0)


# In[ ]:


# Solutions
# Parameters to compare schemes
nt = 100
nx = 40
dt = 1/nt
dx = 1/nx
c = dt*nx
initial = combi
x = np.arange(0, 1, dx)
phi0 = initial(x)
plt.plot(x, phi0, 'k', label = 't=0')

FCT1 = lambda phi, c : FCT(phi, c, nCorr=1)
FCT2 = lambda phi, c : FCT(phi, c, nCorr=2)
FCT3 = lambda phi, c : FCT(phi, c, nCorr=3)

fluxes = [PPMflux, upwindFlux, FCT1, FCT2, FCT3]
names = ['PPM', 'upwind', 'PPM with FCT 1', 'PPM with FCT 2', 'PPM with FCT 3']

for flux, name in zip(fluxes, names):
    phi = phi0.copy()
    for it in range(nt):
        phi = advect(phi, c, flux)
    plt.plot(x, phi, label=name)

plt.legend(bbox_to_anchor=(1.1, 1))
plt.title('Explicit solutions after one revolution\n'
          +'with iterative application of FCT\n'
          + 'c = '+str(round(c,2))+', dt = '+str(round(dt,2))
           +', nt = '+str(nt)+', nx = '+str(nx))
plt.xlim([0,1])
plt.savefig('PPM_FCT.pdf')
plt.show()


# 1D flux-form advection schemes on uniform grids
# These functions provides the weights to calculate face interface values
# from cell values.

# Numerical methods assuming a one-dimensional, uniform, periodic grid
# A periodic domain is implimented using numpy roll.
# Interface j+1/2 is indexed j so that cell j is between interfaces indexed j-1 and j
# Assumes positive velocity so cell j is upwind of interface j-1
#         |  cell |   
#   j-1   |   j   |   j+1
#        j-1      j

import numpy as np

def advect(phi, c, fluxWeights):
    """Advect phi for one time step with Courant number c using flux in 
    function fluxWeights"""
    flux = fluxSum(phi, fluxWeights)
    return phi - c*(flux - np.roll(flux,1))

def fluxSum(phi, fluxWeights):
    fluxHalf = np.zeros_like(phi)
    [indicies, weights] = fluxWeights()
    for j,w in zip(indicies, weights):
        fluxHalf += w*np.roll(phi,-j)
    return fluxHalf

def CD():
    """Returns the indicies and the weights for calculating the flux at
    j+1/2 from surrouning cell values. For use in function fluxSum"""
    indicies = [0,1]
    weights = [0.5, 0.5]
    return indicies, weights

def up():
    """Returns the indicies and the weights for calculating the flux at
    j+1/2 from surrouning cell values. For use in function fluxSum"""
    indicies = [0]
    weights = [1]
    return indicies, weights

def qCubic():
    """Returns the indicies and the weights for calculating the flux at
    j+1/2 from surrouning cell values. For use in function fluxSum"""
    indicies = [-1,0,1]
    weights = [-1/6, 5/6, 1/3]
    return indicies, weights

def linUp():
    """Returns the indicies and the weights for calculating the flux at
    j+1/2 from surrouning cell values. For use in function fluxSum"""
    indicies = [-1,0,1]
    weights = [-1/4, 1, 1/4]
    return indicies, weights


from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import numpy as np

def advect(phi, c, alpha, correction = None):
    """An adveciton scheme to advect phi for one time step with Cournat number
    c and off-centering alpha between forward (alpha=0) and backward (alpha=1).
    correction is the high-order correction on first-order upwind
    phi (1d array): of dependent variable at start time.
    c (float): The uniform Courant number
    alpha (float or callable function): the off-centering
    kmax (int): the number of iterative solves (default 2)
    correction (callable): the function for the high-order correction
                   default = None. Arguments phi, phiOld, c, alpha"""
    a = alpha(c) if callable(alpha) else alpha
    nx = len(phi)
    phiOld = phi.copy()
    phi1 = phiOld - (1-a)*c*(phi - np.roll(phi,1))
    
    # Matrix for implicit part
    M = diags([-a*c*np.ones(nx-1),(1+a*c)*np.ones(nx),-a*c],
                   [-1,0,nx-1], shape=(nx,nx), format = 'csr')

    kmax = 1
    HOcorr = None
    if correction is not None:
        if "kmax" in correction.keys():
            kmax = correction["kmax"]
        HOcorr = correction["HOcorr"]

        # Additional high-order matrix coefficients
        if 'HOmatrix' in correction.keys():
            M += correction["HOmatrix"](nx,c, a)
    
    # Outer iterations per time step
    for k in range(kmax):
        RHS = phi1
        if callable(HOcorr):
            correction["k"] = k
            RHS += HOcorr(phi, phiOld, c, a, correction)
        phi = spsolve(M, RHS)
    return phi

def WB_ImEx_expCorr(phi, phiOld, c, a, options):
    return -0.5*(1-a)*c*(1-c)*(phiOld - 2*np.roll(phiOld,1) + np.roll(phiOld,2))

def WB_ImEx_impMatrix(nx, c, a):
    coeff = 0.5*a*c*(1+c)
    return coeff*diags([np.ones(nx-2), -2*np.ones(nx-1), np.ones(nx),
                        np.ones(2), -2], [-2,-1,0,nx-2,nx-1],
                       shape=(nx,nx), format = 'csr')

def WB_Im_Ex(phi, phiOld, c, a, options):
    """Implicit-explicit version of Warming and Beam.
    A corrector on first-order in space, trapezoidal ImEx in time
    phi (1d array): of dependent variable at start time.
    c (float): The uniform Courant number
    a (float or callable function): the off-centering
    chi (float or callable function): the limit on the correction
    kmax (int): the number of HO corrections"""
    
    k = options["k"]
    chi = options["chi"]
    if callable(chi):
        ch = chi(c, k, options)
    else:
        ch = chi
    
    return -ch*0.5*c*\
    (
        (1-a)*(1-c)*(phiOld - 2*np.roll(phiOld,1) + np.roll(phiOld,2))
      + a*(1+c)*(phi - 2*np.roll(phi,1) + np.roll(phi,2))
    )


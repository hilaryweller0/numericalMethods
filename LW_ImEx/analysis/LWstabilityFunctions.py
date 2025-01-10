import numpy as np
from numpy import exp, pi, cos

def chiFromC(c):
    return np.minimum(np.maximum(2/c-1, 0), 1)

def chiFromAlpha(a):
    return np.minimum(np.maximum(1 - 2*a, 0), 1)

# Maximum magnitude of the amplification factor for the full implicit scheme
def magA_full(c, a, chi):
    """ c is the Courant number
        a is the off-centering
        chi is the temporal high order limiter"""
    kdxs = np.linspace(0, 2*pi, 37)
    maxMag = 0
    for ik in range(len(kdxs)):
        kdx = kdxs[ik]
        A = (1 - (1-a)*c*(1 - exp(-1j*kdx)) - 
                  (1-a)*0.5*c*(1-chi*c)*(exp(1j*kdx) - 2 + exp(-1j*kdx))) \
            / (1 + a*c*(1 - exp(-1j*kdx)) 
                   + a*0.5*c*(1+chi*c)*(exp(1j*kdx) - 2 + exp(-1j*kdx)))
        maxMag = max(maxMag, abs(A))
    return maxMag

# Maximum magnitude of the amplification factor for predictor-corrector scheme
def magA_PC(c, a, chi, kmax):
    """ c is the Courant number
        a is the off-centering
        chi is the temporal high order limiter
        kmax is the number of predictor-corrector iterations"""
    kdxs = np.linspace(0, 2*pi, 36)
    maxMag = 0
    for ik in range(len(kdxs)):
        kdx = kdxs[ik]
        A = 1 + 0*1j
        for k in range(kmax):
            A = (1 - (1-a)*c*(1 - exp(-1j*kdx))
            + c*(1-cos(kdx))*((1-a)*(1-chi*c) + a*(1+chi*c)*A)) \
                   / (1 + a*c*(1 - exp(-1j*kdx)))
        maxMag = max(maxMag, abs(A))
    return maxMag


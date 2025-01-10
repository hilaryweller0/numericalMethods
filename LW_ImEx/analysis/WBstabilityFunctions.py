import numpy as np
from numpy import exp, pi, cos

def chiFromC(c, k):
    chi = 0
    if k == 2:
        chi = np.minimum(np.maximum((2*c-1)/(3*c-3), 0), 1)
    return chi


# Maximum magnitude of the amplification factor for the full implicit scheme
def magA_full(c, a, chi):
    """ c is the Courant number
        a is the off-centering
        chi is the temporal high order limiter"""
    kdxs = np.linspace(0, 2*pi, 37)
    maxMag = 0
    for ik in range(len(kdxs)):
        kdx = kdxs[ik]
        A = A_full(c, a, chi, kdx)
        maxMag = max(maxMag, abs(A))
    return maxMag

def A_full(c, a, chi, kdx):
    return (1 - (1-a)*c*(1 - exp(-1j*kdx))
               - chi*(1-a)*0.5*c*(1-c)*(1 - 2*exp(-1j*kdx) + exp(-2*1j*kdx))) \
            / (1 + a*c*(1 - exp(-1j*kdx)) 
                   + chi*a*0.5*c*(1+c)*(1 - 2*exp(-1j*kdx) + exp(-2*1j*kdx)))

# Maximum magnitude of the amplification factor for predictor-corrector scheme
def magA_PC(c, a, kmax, chi=-1, d=1):
    """ c is the Courant number
        a is the off-centering
        kmax is the number of predictor-corrector iterations
        chi is the temporal high order limiter (default -1 means calculated)"""
    kdxs = np.linspace(0, 2*pi, 36)
    maxMag = 0
    for ik in range(len(kdxs)):
        kdx = kdxs[ik]
        maxMag = max(maxMag, abs(A_PC(c, a, kmax, kdx, chi=chi, d=d)))
    return maxMag

def A_PC(c, a, kmax, kdx, chi=-1, d=1):
    A = 1 + 0*1j
    chiCalc = chi == -1
    for k in range(kmax):
        if chiCalc:
            chi = 0
            if k == 1:
                if c <= 1: #1.5:
                    chi = 1
                else:
                    chi = np.minimum(1,(2*c-1)*(d*(2*c-1)+1)/(6*c*(c-1)))
                    #(2*c-1)/(3*c-3) #(2*c-1)/(6*c*(c-1)) #
        if (k == 0) & (kmax == 2) & (chi == 1) :
            A = (1 - (1-a)*c*(1 - exp(-1j*kdx))) / (1 + a*c*(1 - exp(-1j*kdx)))
        else:
            A = (1 - (1-a)*c*(1 - exp(-1j*kdx))
- chi*0.5*c*((1-a)*(1-c) + a*(1+c)*A)*(1 - 2*exp(-1j*kdx) + exp(-2*1j*kdx))) \
               / (1 + a*c*(1 - exp(-1j*kdx)))
    return A

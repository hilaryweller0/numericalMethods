# Stability analysis of cubic upwind spatial discretisation with RK3 ImEx

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
import os
import sys
import itertools

eps = sys.float_info.epsilon

# Amplification factors
def calcAhc(kdx):
    return 1/6*(2*np.exp(1j*kdx) - 3 + np.exp(-1j*2*kdx))

def calcAL(kdx):
    return 1 - np.exp(-1j*kdx)

def calcA(Ai, Ae, c, kdx):
    # Check that Butcher tableau Ai and Ae are both sxs matrices
    s = len(Ai)
    if (len(np.shape(Ai)) != 2) | (len(np.shape(Ae)) != 2) \
      | (np.shape(Ai)[0] != s) | (np.shape(Ai)[1] != s) \
      | (np.shape(Ae)[0] != s) | (np.shape(Ae)[1] != s):
        raise Exception("In calcA, Butcher tableau Ai and Ae should both be sxs matrices but np.shape(Ai) is", np.shape(Ai), "and np.shape(Ae) is", np.shape(Ae))
    A = np.empty([s], dtype=np.complex128)
    A[0] = 1
    Al = calcAL(kdx)
    Ah = calcAhc(kdx)
    
    for i in range(1,s):
        A[i] = (1 - c*Al*sum(Ai[i,0:i]*A[0:i]) \
             - c*Ah*sum(Ae[i,0:i]*A[0:i]))/(1 + c*Ai[i,i]*Al)
    
    return A[-1]

# Butcher tableau as a function of Courant number
def BA(c, cmax, A1, A2, A3):
    alpha = max(0, 1 - 2*cmax/max(c, eps))
    beta = max(0, 1 - cmax/max(c, eps))
    return (1-beta)*A3 + beta*((1 - alpha)*A2 + alpha*A1)


def calcAllMagA(A1, A2, A3, cs, cmax=1., nkdx = 81, magAkdx = None, kdxs = None):
    if type(magAkdx) == np.ndarray:
        magAkdx.resize([nkdx,len(cs)], refcheck=False)
    if type(kdxs) == np.ndarray:
        kdxs.resize(nkdx, refcheck=False)
    else:
        kdxs = np.zeros(nkdx)
    #kdxs = np.linspace(-np.pi,np.pi, nkdx).copy()
    dk = 2*np.pi/(nkdx-1)
    
    absA = np.zeros([len(cs)])
    for ik in range(len(kdxs)):
        kdx = -np.pi + ik*dk
        kdxs[ik] = kdx
        
        for ic in range(len(cs)):
            c = cs[ic]
            A = calcA(BA(c, cmax, A1, A2, A3), A3, c, kdx)
            if type(magAkdx) == np.ndarray:
                magAkdx[ik,ic] = A
            absA[ic] = max(absA[ic], abs(A))
    return absA

# Implicit and Explicit Butcher tableau
Ai1 = np.array([[0,0,0,0], [0,1,0,0], [0,0,0.5,0], [0,0,0,1]])
Ai2 = np.array([[0,0,0,0], [0.5,0.5,0,0], [0.25,0,0.25,0], [0.5,0,0,0.5]])
Ae3 = np.array([[0,0,0,0], [1,0,0,0], [0.25,0.25,0,0], [1/6,1/6,2/3,0]])
# Range of Courant numbers
cs = np.linspace(0,8,321)

# Magnitude of amplification
kdx = np.array([0.]).copy()
Ak = np.array([[1j]]).copy()
absA_13 = calcAllMagA(Ai1, Ai2, Ae3, cs, cmax=1., magAkdx = Ak, kdxs = kdx)

# Plot max|A| as a function of c
plt.clf()
plt.plot(cs, absA_13, 'k', label='RK3 ImEx')
plt.xlim(0,8)
#plt.ylim(0.99,1.01)
plt.axhline(y=1, ls=(2,(2,10)), color='k', lw=0.25)
plt.axvline(x=1, ls=(2,(2,10)), color='k', lw=0.25)
plt.axvline(x=2, ls=(2,(2,10)), color='k', lw=0.25)
plt.xlabel(r'$c$')
plt.ylabel(r'max $|A|$')
plt.legend(bbox_to_anchor=(-0.01,-0.01), loc='lower left', frameon=False)
plt.tight_layout()
fileName='Acubic_RK3.pdf'
plt.savefig(fileName)
os.system('pdfCrop '+fileName+';evince '+fileName+'&')


# Plot max|A| as a function of c and kdx
plt.clf()
plt.contourf(cs, kdx, abs(Ak), np.arange(0,2,0.1), cmap='bwr',
             extend='both')
plt.colorbar(location='bottom')
plt.contour(cs, kdx, abs(Ak), [1+eps], colors='k')
plt.axvline(x=1, ls=(2,(2,10)), color='k', lw=0.25)
plt.axvline(x=2, ls=(2,(2,10)), color='k', lw=0.25)
plt.xlabel(r'$c$')
plt.ylabel(r'$k\Delta x$')
plt.plot([0], [0], 'k-', label=r'$|A|=1$')
#plt.plot([0], [0], ls=(2,(2,10)), color='k', lw=0.25, label=r'$c=1,2$')
plt.legend()
plt.tight_layout()
fileName='Acubic_RK3_kdx.pdf'
plt.savefig(fileName)
os.system('pdfCrop '+fileName+';evince '+fileName+'&')


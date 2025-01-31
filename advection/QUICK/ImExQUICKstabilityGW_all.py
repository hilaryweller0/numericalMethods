# Stability analysis of QUICK spatial discretisation with RK2 in time and
# Trapezoidal for GW

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
import os
import sys
import itertools

eps = sys.float_info.epsilon

# Different options for alpha

def alpha_05(c, Ndt):
    return 0.5

def alpha_c(c, Ndt):
    return max(0.5, 1-1/max(c, eps))

def alpha_cN(c, Ndt):
    return max(0.5, 1-1/max(c+Ndt, eps))

def alphaSafety_c(c, Ndt):
    return max(0.5, 1-1/max(c+0.2, eps))

def alphaSafety_cN(c, Ndt):
    return max(0.5, 1-1/max(c+Ndt+0.2, eps))

def alphaSafety_cNmax(c, Ndt):
    return max(0.5, 1-1/max(c*(1 + Ndt) + 0.2, eps))

# Amplification factors
def calcA1(kdx):
    return 1 - np.exp(-1j*kdx)

def calcAhc(kdx):
    return 1/8*(3*np.exp(1j*kdx) - 5 + np.exp(-1j*kdx) + np.exp(-1j*2*kdx))

def calcAp(A1, Ahc, alpha, beta, gamma, c, Ndt):
    return (1 - (1-alpha*beta)*c*A1 - gamma*c*Ahc - (1-alpha)*1j*Ndt)\
                     /(1 + alpha*beta*c*A1 + alpha*1j*Ndt)

def calcA(A1, Ahc, Ap, alpha, beta, gamma, c, Ndt):
    return (1 - c*A1*(1 - alpha + alpha*(1 - beta)*Ap)
              - gamma*c*Ahc*(1 - alpha + alpha*Ap)
              - 1j*Ndt*(1 - alpha)) / (1 + alpha*c*beta*A1 + 1j*Ndt*alpha)

def calcAllMagA(Ndts, cs, betas, gammas, alpha, nkdx = 80):
    kdxs = np.linspace(-np.pi,np.pi, nkdx)
    absA = np.zeros([len(Ndts), len(cs), len(betas), len(gammas)])
    for ik in range(len(kdxs)):
        kdx = kdxs[ik]
        A1 = calcA1(kdx)
        Ahc = calcAhc(kdx)
        
        for ic in range(len(cs)):
            c = cs[ic]
            
            for ig in range(len(gammas)):
                g = gammas[ig]
                
                for iN in range(len(Ndts)):
                    Ndt = Ndts[iN]
                    a = alpha(c,Ndt)
                    
                    for ib in range(len(betas)):
                        b = betas[ib]
                    
                        Ap = calcAp(A1, Ahc, a, b, g, c, Ndt)
                        A = calcA(A1, Ahc, Ap, a, b, g, c, Ndt)
                    
                        absA[iN,ic,ib,ig] = max(absA[iN,ic,ib,ig], abs(A))
    return absA

cs = np.linspace(0,4,321) # Range of Courant numbers
gammas = [0,1] # Range of blend for HOC
Ndts = [0]
betas = [0,1]    # ImEx flag (beta=1 for implicit)

absA = calcAllMagA(Ndts, cs, betas, gammas, alpha_c)
absA02 = calcAllMagA(Ndts, cs, betas, gammas, alphaSafety_c)
absA05 = calcAllMagA(Ndts, cs, betas, gammas, alpha_05)

# |A| as a function of c for Ndt=0, gamma=0,1
plt.clf()
plt.plot(cs, absA[0,:, 0,1], 'k', label=r'$\beta=0$, $\gamma=1$, $\alpha=0.5$')
plt.plot(cs, absA[0,:, 0,0], 'k--',label=r'$\beta=0$, $\gamma=0$, $\alpha=0.5$')
plt.plot(cs, absA05[0,:, 1,1], 'r-', label=r'$\beta=1$, $\gamma=1$, $\alpha=0.5$')
plt.plot(cs, absA05[0,:, 1,0], 'r--', label=r'$\beta=1$, $\gamma=0$, $\alpha=0.5$')
plt.plot(cs, absA02[0,:, 1,1], 'b-', label=r'$\beta=1$, $\gamma=1$, $\alpha< 1-1/(c+0.2)$')
plt.plot(cs, absA02[0,:, 1,0], 'b--', label=r'$\beta=1$, $\gamma=0$, $\alpha< 1-1/(c+0.2)$')
plt.xlim(0,4)
plt.ylim(0.99,1.01)
plt.axhline(y=1, ls=(2,(2,10)), color='k', lw=0.25)
plt.axvline(x=0.775, ls=(2,(2,10)), color='k', lw=0.25)
plt.axvline(x=1, ls=(2,(2,10)), color='k', lw=0.25)
plt.xlabel(r'$c$')
plt.ylabel(r'max $|A|$')
plt.legend(bbox_to_anchor=(-0.01,-0.01), loc='lower left', frameon=False)
plt.tight_layout()
fileName='Aquick_N0_gamma_1.pdf'
plt.savefig(fileName)
os.system('pdfCrop '+fileName+';evince '+fileName+'&')

cs = np.linspace(0,5,201) # Range of Courant numbers
gammas = [1] # Range of blend for HOC
Ndts = np.linspace(0, 5, 101)
betas = [1]    # ImEx flag (beta=1 for implicit)

absA = calcAllMagA(Ndts, cs, betas, gammas, alphaSafety_cN)
absA05 = calcAllMagA(Ndts, cs, betas, gammas, alpha_05)
absA_c = calcAllMagA(Ndts, cs, betas, gammas, alphaSafety_c)
absAmax = calcAllMagA(Ndts, cs, betas, gammas, alphaSafety_cNmax)

for (A, fileName) in zip([absA, absA05, absA_c, absAmax], 
                  ['Aquick_gamma_1_beta_1.pdf',
                   'Aquick_gamma_1_beta_1_alpha05.pdf',
                   'Aquick_gamma_1_beta_1_alphac.pdf', 
                   'Aquick_gamma_1_beta_1_alphacN.pdf']):
    plt.clf()
    plt.contourf(cs, Ndts, A[:,:,0,0], np.arange(0.45,1.65,0.1), cmap='bwr',
                 extend='both')
    plt.colorbar(location='bottom')
    plt.contour(cs, Ndts, A[:,:,0,0], [1+eps], colors='k')
    plt.xlabel(r'$c$')
    plt.ylabel(r'$N\Delta t$')
    plt.plot([0], [0], 'k-', label=r'$|A|=1$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fileName)
    os.system('pdfCrop '+fileName+';evince '+fileName+'&')


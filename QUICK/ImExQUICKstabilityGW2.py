# Stability analysis of QUICK spatial discretisation with RK2 in time and
# Trapezoidal for GW

import numpy as np
import matplotlib.pyplot as plt
import os

def alpha(c):
    return np.maximum(0.5, 1-1/np.maximum(c, 1e-12))

cs = np.linspace(0,8,41) # Range of Courant numbers
gammas = np.array([0,1]) # Range of blend for HOC
kdxs = np.linspace(-np.pi,np.pi, 60)
Ndts = np.linspace(0, 5, 51)

absA = np.zeros([len(Ndts), len(cs), len(gammas)])

for ik in range(len(kdxs)):
    kdx = kdxs[ik]
    A1 = 1 - np.exp(-1j*kdx)
    Ahc = 1/8*(3*np.exp(1j*kdx) - 5 + np.exp(-1j*kdx) + np.exp(-1j*2*kdx))
    
    for ic in range(len(cs)):
        c = cs[ic]
        a = alpha(c)
        ag = a
        
        for ig in range(len(gammas)):
            g = gammas[ig]
            
            for iN in range(len(Ndts)):
                iNdt = 1j*Ndts[iN]
                
                Ap = (1 - (1-a)*c*A1 - g*c*Ahc - (1-ag)*iNdt)\
                     /(1 + a*c*A1 + ag*iNdt)
                A = (1 - (1-a)*c*A1 - (1-a)*c*g*Ahc - a*c*g*Ap*Ahc - (1-ag)*iNdt)\
                    /(1 + a*c*A1 + ag*iNdt)
        
                absA[iN,ic,ig] = max(absA[iN,ic,ig], abs(A))

for ig in range(len(gammas)):
    g = gammas[ig]
    plt.clf()
    plt.contourf(cs, Ndts, absA[:,:,ig], np.arange(0.525,1.5,0.05), cmap='bwr',
                extend='both')
    plt.colorbar(location='bottom')
    plt.contour(cs, Ndts, absA[:,:,ig], [1], colors='k')
    plt.xlabel(r'$c$')
    plt.ylabel(r'$N\Delta t$')
    plt.twinx()
    plt.plot([0], [0], 'k-', label=r'$|A|=1$')
    plt.plot(cs, alpha(cs), 'k--', label=r'$\alpha$')
    plt.ylim([0.4,1])
    plt.legend()
    plt.ylabel(r'$\alpha$')
    plt.tight_layout()
    fileName='A_gamma_'+str(g)+'.pdf'
    plt.savefig(fileName)
    os.system('pdfCrop '+fileName+';evince '+fileName+'&')


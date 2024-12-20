# Stability analysis of QUICK spatial discretisation with RK2 in time and
# Trapezoidal for GW

import numpy as np
import matplotlib.pyplot as plt
import os

def alpha(c):
    return np.maximum(0.5, 1-1/np.maximum(c, 1e-12))

cs = np.linspace(0,20,81) # Range of Courant numbers
gammas = np.arange(0, 1.1, 1) # Range of blend for HOC
kdxs = np.linspace(-np.pi,np.pi, 40)
alphags = np.linspace(0.5, 1, 26)
Ndts = np.arange(0, 2.5, .5)

absA = np.zeros([len(gammas), len(Ndts), len(cs), len(alphags)])

for ik in range(len(kdxs)):
    kdx = kdxs[ik]
    A1 = 1 - np.exp(-1j*kdx)
    Ahc = 1/8*(3*np.exp(1j*kdx) - 5 + np.exp(-1j*kdx) + np.exp(-1j*2*kdx))
    
    for ic in range(len(cs)):
        c = cs[ic]
        a = alpha(c)
        
        for ig in range(len(gammas)):
            g = gammas[ig]
            
            for iN in range(len(Ndts)):
                iNdt = 1j*Ndts[iN]
                
                for iag in range(len(alphags)):
                    ag = alphags[iag]
            
                    Ap = (1 - (1-a)*c*A1 - g*c*Ahc - (1-ag)*iNdt)\
                         /(1 + a*c*A1 + ag*iNdt)
                    A = (1 - (1-a)*c*A1 - (1-a)*c*g*Ahc - a*c*g*Ap*Ahc - (1-ag)*iNdt)\
                        /(1 + a*c*A1 + ag*iNdt)
            
                    absA[ig,iN,ic,iag] = max(absA[ig,iN,ic,iag], abs(A))

plt.rcParams['figure.figsize'] = [8, 3]
for iN in range(1,len(Ndts)):
    Ndt = Ndts[iN]
    for ig in range(len(gammas)):
        plt.clf()
        g = gammas[ig]
        absAt = absA[ig,iN,:,:].transpose()
        plt.contourf(cs, alphags, absAt, np.arange(0.5125,1.5,0.025), cmap='bwr',
                    extend='both')
        plt.colorbar()
        plt.contour(cs, alphags, absAt, [1])
        plt.plot(cs, alpha(cs))
        plt.xlabel('c')
        plt.ylabel(r'$\alpha_g$')
        title='$N\Delta t = ' + str(Ndt) + ',\ \gamma = ' + str(g) + '$'
        plt.title(title)
        plt.tight_layout()
        fileName='A_Ndt_'+str(Ndt)+'_gamma_'+str(g)+'.pdf'
        plt.savefig(fileName)
        os.system('pdfCrop '+fileName+';evince '+fileName+'&')


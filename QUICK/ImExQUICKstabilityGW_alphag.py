# Stability analysis of QUICK spatial discretisation with RK2 in time and
# Trapezoidal for GW

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def alpha(c):
    return max(0.5, 1-1/max(c, 1e-12))

def maxAbsA(cs, Ndts, alphags, gamma, N = 40):
    absA = np.zeros([len(Ndts), len(cs), len(alphags)])
    kdxs = np.linspace(-np.pi,np.pi, N)
    g = gamma
    for ik in range(len(kdxs)):
        kdx = kdxs[ik]
        A1 = 1 - np.exp(-1j*kdx)
        Ahc = 1/8*(3*np.exp(1j*kdx) - 5 + np.exp(-1j*kdx) + np.exp(-1j*2*kdx))
        
        for ic in range(len(cs)):
            c = cs[ic]
            a = alpha(c)
            
            for iN in range(len(Ndts)):
                iNdt = 1j*Ndts[iN]
                
                for iag in range(len(alphags)):
                    ag = alphags[iag]
            
                    Ap = (1 - (1-a)*c*A1 - g*c*Ahc - (1-ag)*iNdt)\
                         /(1 + a*c*A1 + ag*iNdt)
                    A = (1 - (1-a)*c*A1 - (1-a)*c*g*Ahc - a*c*g*Ap*Ahc - (1-ag)*iNdt)\
                        /(1 + a*c*A1 + ag*iNdt)
            
                    absA[iN,ic,iag] = max(absA[iN,ic,iag], abs(A))
    return absA

def minAlphag(absA, alphags):
    n0 = np.shape(absA)[0]
    n1 = np.shape(absA)[1]
    n2 = np.shape(absA)[2]
    alphag = 2*np.ones([n0,n1])
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2-1,-1,-1):
                if absA[i0,i1,i2] < 1 + sys.float_info.epsilon:
                        alphag[i0,i1] = alphags[i2]
                else:
                    break
    return alphag

# Calculations for gamma = 0
gamma = 0
cs0 = np.linspace(1,8, 36)  # Range of Courant numbers
Ndts0 = np.linspace(0, 4, 41)
alphags = np.linspace(0.5, 1, 51)
absA0 = maxAbsA(cs0, Ndts0, alphags, gamma)
alphag0 = minAlphag(absA0, alphags)

# Plot for gamma = 0
plt.clf()
plt.contourf(cs0, Ndts0, alphag0, np.arange(0.5,1,0.025), cmap='jet')
plt.colorbar()
plt.xlabel('c')
plt.ylabel(r'$N\Delta t$')
title='$\gamma = ' + str(gamma) + '$'
plt.title(title)
plt.tight_layout()
fileName='alphag_gamma_'+str(gamma)+'.pdf'
plt.savefig(fileName)
os.system('pdfCrop '+fileName+';evince '+fileName+'&')

# Calculations for gamma = 1
gamma = 1
cs1 = np.linspace(0,2,101)  # Range of Courant numbers
Ndts1 = np.linspace(0, 2, 81)
alphags = np.linspace(0.5, 1, 21)
absA1 = maxAbsA(cs1, Ndts1, alphags, gamma)
alphag1 = minAlphag(absA1, alphags)

# Plot for gamma = 1
plt.clf()
plt.contourf(cs1, Ndts1, alphag1, np.arange(0.5,1,0.025), cmap='jet')
plt.colorbar()
plt.xlabel('c')
plt.ylabel(r'$N\Delta t$')
title='$\gamma = ' + str(gamma) + '$'
plt.title(title)
plt.tight_layout()
fileName='alphag_gamma_'+str(gamma)+'.pdf'
plt.savefig(fileName)
os.system('pdfCrop '+fileName+';evince '+fileName+'&')

# Appears to be unstable for c=1, Ndt=0.5, gamma=1
alphags = np.linspace(0.5, 1, 51)
kdxs = np.linspace(-np.pi,np.pi, 40)
absA = np.zeros([len(kdxs), len(alphags)])
g = 1
c = 1
iNdt = 1j*0.5
a = alpha(c)
for ik in range(len(kdxs)):
    kdx = kdxs[ik]
    A1 = 1 - np.exp(-1j*kdx)
    Ahc = 1/8*(3*np.exp(1j*kdx) - 5 + np.exp(-1j*kdx) + np.exp(-1j*2*kdx))
    
    for iag in range(len(alphags)):
        ag = alphags[iag]
        Ap = (1 - (1-a)*c*A1 - g*c*Ahc - (1-ag)*iNdt)\
             /(1 + a*c*A1 + ag*iNdt)
        A = (1 - (1-a)*c*A1 - (1-a)*c*g*Ahc - a*c*g*Ap*Ahc - (1-ag)*iNdt)\
            /(1 + a*c*A1 + ag*iNdt)
        absA[ik,iag] = abs(A)
plt.clf()
plt.contourf(kdxs, alphags, absA.transpose(), np.arange(0.5125,1.5,0.025),
             cmap='bwr', extend='both')
plt.colorbar()
plt.contour(kdxs, alphags, absA.transpose(), [1])
plt.xlabel(r'$k\Delta x$')
plt.ylabel(r'$\alpha_g$')
plt.tight_layout()
fileName='A_c1Ndt05g1.pdf'
plt.savefig(fileName)
os.system('pdfCrop '+fileName+';evince '+fileName+'&')


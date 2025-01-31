# Stability analysis of QUICK spatial discretisation with RK2 in time.

import numpy as np
import matplotlib.pyplot as plt

def alpha(c):
    return max(0.5, 1-1/max(c, 1e-12))

cs = np.linspace(0,4,81) # Range of Courant numbers
gammas = np.linspace(0, 1, 41) # Range of blend for HOC
kdxs = np.linspace(-np.pi,np.pi, 80)
absA = np.zeros([len(cs), len(gammas)])
Ag0 = np.zeros([len(cs), len(kdxs)])
Ag1 = np.zeros([len(cs), len(kdxs)])

for ik in range(len(kdxs)):
    kdx = kdxs[ik]
    A1 = 1 - np.exp(-1j*kdx)
    Ahc = 1/8*(3*np.exp(1j*kdx) - 5 + np.exp(-1j*kdx) + np.exp(-1j*2*kdx))
    
    for ic in range(len(cs)):
        c = cs[ic]
        a = alpha(c)
        
        for ig in range(len(gammas)):
            g = gammas[ig]
            Ap = (1 - (1-a)*c*A1 - g*c*Ahc)/(1 + a*c*A1)
            A = (1 - (1-a)*c*A1 - (1-a)*c*g*Ahc - a*c*g*Ap*Ahc)\
                /(1 + a*c*A1)
            
            absA[ic,ig] = max(absA[ic,ig], abs(A))
        
        g = 0
        Ap = (1 - (1-a)*c*A1 - g*c*Ahc)/(1 + a*c*A1)
        Ag0[ic,ik] = abs((1 - (1-a)*c*A1 - (1-a)*c*g*Ahc - a*c*g*Ap*Ahc)\
                        /(1 + a*c*A1))
        g = 1
        Ap = (1 - (1-a)*c*A1 - g*c*Ahc)/(1 + a*c*A1)
        Ag1[ic,ik] = abs((1 - (1-a)*c*A1 - (1-a)*c*g*Ahc - a*c*g*Ap*Ahc)\
                        /(1 + a*c*A1))
        

print(np.amax(absA))
plt.contourf(cs, gammas, absA.transpose())
plt.colorbar()
plt.contour(cs, gammas, absA.transpose(), [1])
plt.show()

print(np.amax(Ag1))
plt.contourf(cs, kdxs, Ag1.transpose())
plt.colorbar()
plt.contour(cs, kdxs, Ag1.transpose(), [1])
plt.show()

print(np.amax(Ag0))
plt.contourf(cs, kdxs, Ag0.transpose())
plt.colorbar()
plt.contour(cs, kdxs, Ag0.transpose(), [1])
plt.show()


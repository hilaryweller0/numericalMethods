# Stability analysis of QUICK spatial discretisation with RK2 in time.

import numpy as np
import matplotlib.pyplot as plt

alphas = np.linspace(0.4,1,61)  # off centering
cs = np.linspace(0.7,1.1,41) # Range of Courant numbers
kdxs = np.linspace(-np.pi,np.pi, 80)
absA = np.zeros([len(cs), len(alphas)])
As = np.zeros([len(cs), len(kdxs)], complex)
for ic in range(len(cs)):
    c = cs[ic]
    for ik in range(len(kdxs)):
        kdx = kdxs[ik]
        Am = 3/4*np.exp(-1j*kdx) + 3/8 - 1/8*np.exp(-1j*2*kdx)
        Ap = 3/4 + 3/8*np.exp(1j*kdx) - 1/8*np.exp(-1j*kdx)
        App = 1 - c*(Ap - Am)
        for ia in range(len(alphas)):
            alpha = alphas[ia]
            A = 1 - c*(Ap - Am)*(1 - alpha + alpha*App)
            absA[ic,ia] = max(absA[ic,ia], abs(A))
        As[ic,ik] = 1 - 0.5*c*(Ap - Am)*(1 + App)

plt.contourf(alphas, cs, absA)
plt.colorbar()
plt.contour(alphas, cs, absA, [1])
plt.show()

plt.contourf(kdxs, cs, abs(As))
plt.colorbar()
plt.contour(kdxs, cs, abs(As), [1])
plt.show()

from schemes import *
from helpers import *
import numpy as np
import matplotlib.pyplot as plt

# Plot convergence where only spatial resolution changes
# Constant parameters
nt = 100
dt = 0.01
nxs = np.array([10,20,40,80,120,160,180,220,300,400,600,1000])
init = cosine

schemes = [{"HOcorr": WB_Im_Ex, "chi": chiFromcd, "kmax": 2, "damp": 0,
            "filename": "WBPC_ImEx_d0", "alpha": alphaFromc},
           {"HOcorr": WB_ImEx_expCorr, "HOmatrix": WB_ImEx_impMatrix,
            "filename": "WB_ImEx", "alpha": alphaFromc},
           {"HOcorr": WB_ImEx_expCorr, "HOmatrix": WB_ImEx_impMatrix,
            "filename": "WB_ImEx_a2", "alpha": alphaFromcWB},
           {"HOcorr": WB_ImEx_expCorr, "HOmatrix": WB_ImEx_impMatrix,
            "filename": "WB_ImEx_a1", "alpha": 1},
           {"HOcorr": WB_ImEx_expCorr, "HOmatrix": WB_ImEx_impMatrix,
            "filename": "WB_ImEx_a05", "alpha": 0.5}]

cs = dt*nxs
fig, ax1 = plt.subplots()
plt.xlabel(r'$c$')
plt.ylabel(r'$\ell_2$')
ax2 = ax1.twinx()  
ax2.semilogx(cs, alphaFromcWB(cs),'k:', label=r'$\alpha=0.5(c-1.9)/c$')
ax2.semilogx(cs, alphaFromc(cs),'k--', label=r'$\alpha=1-1/c$')
ax2.legend()
plt.ylabel(r'$\alpha$')
for scheme in schemes:
    l2s = np.zeros(len(nxs))
    for i in range(len(nxs)):
        nx = nxs[i]
        dx = 1/nx
        x = np.arange(0, 1, dx)
        phi0 = init(x)
        c = cs[i]
        print('c =', round(c,2), 'dx =', dx, 'nx =', nx)
        phi = phi0.copy()
        
        # Loop through all time steps
        for it in range(nt):
            phi = advect(phi, c, scheme["alpha"], correction = scheme)
        
        l2s[i] = np.sqrt(np.sum((phi - phi0)**2)/np.sum(phi0**2))
    ax1.loglog(cs, l2s, label=scheme["filename"])

ax1.legend()
saveFig("l2errorsVdx")


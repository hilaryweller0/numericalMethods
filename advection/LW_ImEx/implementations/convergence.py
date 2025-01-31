from schemes import *
from helpers import *
import numpy as np
import matplotlib.pyplot as plt

schemes = [{"HOcorr": WB_Im_Ex, "chi": chiFromcd, "kmax": 2, "damp": 0,
            "filename": "WBPC_ImEx_d0", "alpha": alphaFromc},
           {"HOcorr": WB_ImEx_expCorr, "HOmatrix": WB_ImEx_impMatrix,
            "filename": "WB_ImEx", "alpha": alphaFromc},
           {"HOcorr": WB_ImEx_expCorr, "HOmatrix": WB_ImEx_impMatrix,
            "filename": "WB_ImEx_a2", "alpha": alphaFromcWB},
           {"HOcorr": WB_ImEx_expCorr, "HOmatrix": WB_ImEx_impMatrix,
            "filename": "WB_ImEx_a1", "alpha": 1}]

# Convergence with resolution
init = cosine
nxs = np.array([32,64,128,256])
dxs = 1/nxs
plt.loglog(dxs, 100*dxs, 'k:', label='1st/2nd order')
plt.loglog(dxs, 100*dxs**2, 'k:')
for scheme in schemes:
    l2s = np.zeros(len(dxs))
    #for c in [0.4,0.8,1.6, 2,3.2,6.4]:
    for c in [1.6,3.2]:
        for i in range(len(dxs)):
            dx = dxs[i]
            nx = nxs[i]
            x = np.arange(0, 1, dx)
            phi0 = init(x)
            dt = c*dx
            nt = int(nx/c)
            endTime = dt*nt
            print('End time =', round(endTime,3), 'dt =', round(dt,4),
                 'c =', c, 'dx =', dx, 'nx =', nx, 'nt =', nt)
            phi = phi0.copy()

            # Loop through all time steps
            for it in range(nt):
                phi = advect(phi, c, scheme["alpha"], correction = scheme)

            l2s[i] = np.sqrt(np.sum((phi - phi0)**2)/np.sum(phi0**2))
        plt.loglog(dxs, l2s, 
                   label=scheme["filename"]+', c = '+str(round(c,2)))

plt.legend()
plt.xlabel(r'$\Delta x$')
plt.ylabel(r'$\ell_2$')
saveFig("l2errorsVdxdt")


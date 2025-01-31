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
            "filename": "WB_ImEx_a1", "alpha": 1},
           {"HOcorr": WB_ImEx_expCorr, "HOmatrix": WB_ImEx_impMatrix,
            "filename": "WB_ImEx_a05", "alpha": 0.5}]

# Constant parameters
nt = 100
dt = 0.01
nxs = np.array([50,80,120,200,400])
init = squareWave

for scheme in schemes:
    nx = nxs[-1]
    x = np.arange(0,1,1/nx)
    plt.plot(x, init(x), 'k', label='t=0')
    print("Scheme", scheme["filename"])
    for nx in nxs:
        dx = 1/nx
        x = np.arange(0, 1, dx)
        phi0 = init(x)
        c = dt/dx
        endTime = dt*nt
        print('End time =', endTime, 'dt =', dt, 'c =', c, 'dx =', dx)
        phi = phi0.copy()

        # Loop through all time steps
        for it in range(nt):
            phi = advect(phi, c, scheme["alpha"], correction = scheme)

        # Plot final time
        l2error = np.sqrt(np.sum((phi - phi0)**2)/np.sum(phi0**2))
        label = 'nx='+str(nx)+', c = '+str(round(c,2))\
              +', l2 = '+str(round(l2error,3))
        print("Scheme", scheme["filename"], "l2error =", l2error)
        plt.plot(x, phi, label=label)

    plt.legend()
    plt.xlabel('x')
    saveFig(scheme["filename"])


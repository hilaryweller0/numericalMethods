import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from WBstabilityFunctions import *
from stabilityFunctions import *

# Constants for all plots
cs = 10**(np.linspace(-1, 1, 81))
twoI = int(np.round(1.3/(2/80)))
cs[twoI] = 2
alphas = np.linspace(0, 1, 51)
chis = np.linspace(0,1,41)
kmax = 2

# for colourscale
levels=np.arange(0.9, 1.9, 0.1)
cnorm = colors.BoundaryNorm(levels, 150)

# For |A| for a range of kdx and c for chi=1
kdxs = np.linspace(0, 2*pi, 37)
magA = np.zeros([len(kdxs), len(cs)])
ReA = np.zeros([len(kdxs), len(cs)])
for kmax in [1,2]:
    for ic in range(len(cs)):
        c = cs[ic]
        a = alphaFromc(c)
        for ik in range(len(kdxs)):
            kdx = kdxs[ik]
            magA[ik,ic] = abs(A_PC(c, a,  kmax, kdx, chi=1-1e-11))
            ReA[ik,ic] = A_PC(c, a,  kmax, kdx, chi=1-1e-11).real
    
    plt.contourf(cs, kdxs,magA)#, levels, norm=cnorm,
#                 cmap='Greys', extend='both')
    plt.colorbar()
    plt.contour(cs, kdxs, magA, [0, 1], colors=['k', 'k'])
    plt.ylabel(r'$k\Delta x$')
    plt.xlabel(r'$c$')
    saveFig("WBPCmagA_kdx_c_chi1_k"+str(kmax))
    
    plt.contourf(cs, kdxs, ReA)#, levels, norm=cnorm,
#                 cmap='Greys', extend='both')
    plt.colorbar()
    plt.contour(cs, kdxs, ReA, [-1, 1], colors=['k', 'k'])
    plt.ylabel(r'$k\Delta x$')
    plt.xlabel(r'$c$')
    saveFig("WBPC_ReA_kdx_c_chi1_k"+str(kmax))

plt.semilogx(cs, alphaFromc(cs), 'k', label=r'$\alpha$')
plt.semilogx(cs, np.where(cs > 2, (2*cs-1)/(3*cs-3), 1), 'k--', label=r'$\chi$')
plt.semilogx(cs, np.where(cs > 1.5, (2*cs-1)/(6*cs*(cs-1)), 1), 'k--', label=r'$\chi_0$')
plt.legend()
plt.xlabel(r'$c$')
plt.axvline(x=1, ls=':', c='k', lw=0.5)
plt.axvline(x=2, ls=':', c='k', lw=0.5)
plt.axhline(y=2/3, ls=':', c='k', lw=0.5)
saveFig("WBPC_alpha_chi")


# Calculate A for WB2_ImEx_pCorr (predictor-corrector, kmax iterations)
maxMagA = np.zeros([kmax, len(cs)])
for k in range(kmax):
    for ic in range(len(cs)):
        c = cs[ic]
        a = alphaFromc(c)
        maxMagA[k,ic] = magA_PC(c, a, k+1)

# Plot max(mag(A)) for each k with calcualted chi
plt.semilogx(cs, maxMagA[0,:], 'k', label=r'k=1')
plt.semilogx(cs, maxMagA[1,:], 'b', label=r'k=2')
plt.legend()
saveFig("WBPCmagA_c_k")

# For |A| for a range of kdx and c for predecitor-corrector version with calculated chi
kdxs = np.linspace(0, 2*pi, 37)
magA = np.zeros([len(kdxs), len(cs)])
ReA = np.zeros([len(kdxs), len(cs)])
kmax=2
for chi in [1,-1]:
    for ic in range(len(cs)):
        c = cs[ic]
        a = alphaFromc(c)
        for ik in range(len(kdxs)):
            kdx = kdxs[ik]
            magA[ik,ic] = abs(A_PC(c, a,  kmax, kdx, chi=chi))
            ReA[ik,ic] = A_PC(c, a,  kmax, kdx, chi=chi).real
    
    plt.contourf(cs, kdxs,magA)#, levels, norm=cnorm,
#                 cmap='Greys', extend='both')
    plt.colorbar()
    plt.contour(cs, kdxs, magA, [0, 1], colors=['k', 'k'])
    plt.ylabel(r'$k\Delta x$')
    plt.xlabel(r'$c$')
    saveFig("WBPCmagA_kdx_c_k"+str(kmax)+"chi_"+str(chi))
    
    plt.contourf(cs, kdxs, ReA)#, levels, norm=cnorm,
#                 cmap='Greys', extend='both')
    plt.colorbar()
    plt.contour(cs, kdxs, ReA, [-1, 1], colors=['k', 'k'])
    plt.ylabel(r'$k\Delta x$')
    plt.xlabel(r'$c$')
    saveFig("WBPC_ReA_kdx_c_k"+str(kmax)+"chi_"+str(chi))

# |A| for a range of kdx and c for predecitor-corrector version with calculated chi using different damping, d
kdxs = np.linspace(0, 2*pi, 37)
magA = np.zeros([len(kdxs), len(cs)])
ReA = np.zeros([len(kdxs), len(cs)])
kmax=2
chi = -1
for d in [0, 0.25, 0.5]:
    for ic in range(len(cs)):
        c = cs[ic]
        a = alphaFromc(c)
        for ik in range(len(kdxs)):
            kdx = kdxs[ik]
            magA[ik,ic] = abs(A_PC(c, a,  kmax, kdx, chi=chi, d=d))
            ReA[ik,ic] = A_PC(c, a,  kmax, kdx, chi=chi,d=d).real
    
    plt.contourf(cs, kdxs,magA)#, levels, norm=cnorm,
#                 cmap='Greys', extend='both')
    plt.colorbar()
    plt.contour(cs, kdxs, magA, [0, 1], colors=['k', 'k'])
    plt.ylabel(r'$k\Delta x$')
    plt.xlabel(r'$c$')
    saveFig("WBPCmagA_kdx_c_k"+str(kmax)+"d_"+str(d))


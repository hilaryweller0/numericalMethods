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

# x-y graphs of fully implicit scheme
plt.semilogx(cs, [magA_full(c, alphaFromc(c), 1) for c in cs], 'k', 
            label=r'$\alpha=1-1/c,\ \chi=1$')
plt.semilogx(cs, [magA_full(c, 0.5, 1) for c in cs], 'b', 
            label=r'$\alpha=0.5,\ \chi=1$')
plt.semilogx(cs, [magA_full(c, 0, 1) for c in cs], 'r', 
            label=r'$\alpha=0,\ \chi=1$')
plt.legend()
plt.ylim([0.95,1.3])
plt.xlabel(r'$c$')
plt.ylabel(r'max $|A|$')
saveFig("WBmagA_full")

# magA for fully implicit with various chi
for chi in [0,1]:
  maxMagA = np.zeros([len(alphas), len(cs)])
  for ia in range(len(alphas)):
    for ic in range(len(cs)):
        maxMagA[ia,ic] = magA_full(cs[ic], alphas[ia], chi)
  
  # Plot max(mag(A))
  plt.semilogx(cs, np.maximum(1-1/cs, 0), 'k--', label=r'$\alpha=\max(0,1-1/c)$')
  plt.semilogx(cs, np.maximum(0, np.minimum(1, (cs-1.9)/(2*cs))),
   'k:', label=r'$\alpha= (c-1.9)/(2*c)$')
  plt.legend()
  plt.axvline(x=1, ls=':', c='k', lw=0.5)
  plt.axvline(x=2, ls=':', c='k', lw=0.5)
  plt.axhline(y=0.5, ls=':', c='k', lw=0.5)
  plt.contourf(cs, alphas, maxMagA, levels, norm=cnorm,
               cmap='Greys', extend='both')
  plt.colorbar()
  plt.contour(cs, alphas, maxMagA, [0, 1+1e-15], colors=['k', 'k'])
  plt.xlabel(r'$c$')
  plt.ylabel(r'$\alpha$')
  saveFig("WBmagA_c_alpha_chi_"+str(chi))

# Full magA for alpha = 1-1/c for a range of chi
maxMagA = np.zeros([len(chis), len(cs)])
for i in range(len(chis)):
    for ic in range(len(cs)):
        c = cs[ic]
        a = max(0, 1-1/c)
        maxMagA[i,ic] = magA_full(c, a, chis[i])

# Plot max(mag(A))
#plt.semilogx(cs, chiFromC(cs), 'k--', label=r'$\chi = 1 -2\alpha$')
plt.legend()
plt.axvline(x=1, ls=':', c='k', lw=0.5)
plt.axvline(x=2, ls=':', c='k', lw=0.5)
plt.axhline(y=0.5, ls=':', c='k', lw=0.5)
plt.contourf(cs, chis, maxMagA, levels, norm=cnorm,
           cmap='Greys', extend='both')
plt.colorbar()
plt.contour(cs, chis, maxMagA, [0, 1+1e-15], colors=['k', 'k'])
plt.xlabel(r'$c$')
plt.ylabel(r'$\chi$')
saveFig("WBmagA_c_chi")


# |A| for a range of kdx and c for calcualted alpha as 1-1/c
kdxs = np.linspace(0, 2*pi, 37)
chi = 1
def alphaFromc2(c):
    return np.maximum(0, np.minimum(1, (0.5+1e-6)*(c-2)/c))

def alphaFromc3(c):
    return np.maximum(0, np.minimum(1, 0.5*(c-1.9)/c))

magA = np.zeros([len(kdxs), len(cs)])
atype=1
for alpha in [alphaFromc,alphaFromc2,alphaFromc3]:
    for ic in range(len(cs)):
        c = cs[ic]
        a = alpha(c)
        for ik in range(len(kdxs)):
            kdx = kdxs[ik]
            magA[ik,ic] = abs(A_full(c, a, chi, kdx))
    
    plt.contourf(cs, kdxs,magA)
    plt.colorbar()
    plt.contour(cs, kdxs, magA, [0, 1], colors=['k', 'k'])
    plt.ylabel(r'$k\Delta x$')
    plt.xlabel(r'$c$')
    saveFig("WBmagA_kdx_c_a"+str(atype))
    atype += 1


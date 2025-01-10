import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from LWstabilityFunctions import *
from stabilityFunctions import *

# Constants for all plots
cs = 10**(np.linspace(-1, 1, 81))
twoI = int(np.round(1.3/(2/80)))
cs[twoI] = 2
alphas = np.linspace(0, 1, 51)
chis = np.linspace(0,1,41)

# for colourscale
levels=np.arange(0.9, 1.9, 0.1)
cnorm = colors.BoundaryNorm(levels, 150)

# x-y graphs of fully implicit scheme
plt.semilogx(cs, [magA_full(c, 0.5, 1) for c in cs], 'k', 
            label=r'$\alpha=0.5,\ \chi=1$')
plt.semilogx(cs, [magA_full(c, 0.5, 0) for c in cs], 'k--', 
            label=r'$\alpha=0.5,\ \chi=0$')
plt.semilogx(cs, [magA_full(c, 0, 1) for c in cs], 'r', 
            label=r'$\alpha=0,\ \chi=1$')
plt.semilogx(cs, [magA_full(c, 0, 0) for c in cs], 'r--', 
            label=r'$\alpha=0,\ \chi=0$')
plt.semilogx(cs, [magA_full(c, 1, 1) for c in cs], 'b', 
            label=r'$\alpha=1,\ \chi=1$')
plt.semilogx(cs, [magA_full(c, 1, 0) for c in cs], 'b--', 
            label=r'$\alpha=1,\ \chi=0$')
plt.legend()
plt.ylim([0.95,1.3])
plt.xlabel(r'$c$')
plt.ylabel(r'max $|A|$')
saveFig("magA_full")

# magA for fully implicit with various chi
for chi in [0,0.5,1]:
  maxMagA = np.zeros([len(alphas), len(cs)])
  for ia in range(len(alphas)):
    for ic in range(len(cs)):
        maxMagA[ia,ic] = magA_full(cs[ic], alphas[ia], chi)
  
  # Plot max(mag(A))
  plt.semilogx(cs, np.maximum(1-1/cs, 0), 'k--', label=r'$\alpha=\max(0,1-1/c)$')
  plt.semilogx(cs, np.maximum(1-1/cs**2, 0), 'k.-', label=r'$\alpha=\max(0,1-1/c^2)$')
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
  saveFig("magA_c_alpha_chi_"+str(chi))

# Full magA for alpha = 1-1/c for a range of chi
maxMagA = np.zeros([len(chis), len(cs)])
for i in range(len(chis)):
    for ic in range(len(cs)):
        c = cs[ic]
        a = max(0, 1-1/c)
        maxMagA[i,ic] = magA_full(c, a, chis[i])

# Plot max(mag(A))
plt.semilogx(cs, chiFromC(cs), 'k--', label=r'$\chi = 1 -2\alpha$')
plt.legend()
plt.axvline(x=1, ls=':', c='k', lw=0.5)
plt.axvline(x=2, ls=':', c='k', lw=0.5)
plt.axhline(y=0.5, ls=':', c='k', lw=0.5)
plt.contourf(cs, chis, maxMagA, levels, norm=cnorm,
           cmap='Greys', extend='both')
plt.colorbar()
plt.contour(cs, chis, maxMagA, [0, 1], colors=['k', 'k'])
plt.xlabel(r'$c$')
plt.ylabel(r'$\chi$')
saveFig("magA_c_chi")


# Calculate A for LW2_ImEx_pCorr (predictor-corrector, kmax iterations)
kmax = 2
maxMagA = np.zeros([kmax, len(chis), len(cs)])

for k in range(kmax):
  for i in range(len(chis)):
    chi = chis[i]
    for ic in range(len(cs)):
      c = cs[ic]
      a = max(0, 1-1/c)
      maxMagA[k,i,ic] = magA_PC(c, a, chi, k+1)


# Plot max(mag(A)) for each k
for k in range(kmax):
    plt.semilogx(cs, np.maximum(1-1/cs, 0), 'k--', label=r'$\alpha=\max(0,1-1/c)$')
    plt.semilogx(cs, np.minimum(np.maximum(2/cs-1, 0), 1), 'k:', label=r'$\chi=1-2\alpha$')
    plt.legend()
    plt.axvline(x=1, ls=':', c='k', lw=0.5)
    plt.axvline(x=2, ls=':', c='k', lw=0.5)
    plt.axhline(y=0.5, ls=':', c='k', lw=0.5)
    plt.contourf(cs, chis, maxMagA[k,:,:], levels, norm=cnorm,
                 cmap='Greys', extend='both')
    plt.colorbar()
    plt.contour(cs, chis, maxMagA[k,:,:], [0, 1], colors=['k', 'k'])
    plt.xlabel(r'$c$')
    plt.ylabel(r'$\chi$')
    saveFig("magA_c_k_"+str(k+1))

# Specific for chi=0/1
maxMagA = np.zeros([len(alphas), len(cs)])
for k in range(kmax):
    for ia in range(len(alphas)):
        a = alphas[ia]
        for ic in range(len(cs)):
            c = cs[ic]
            chi = (c <= 1)
            maxMagA[ia,ic] = magA_PC(c, a, chi, k+1)
    
    # Plot max(mag(A))
    plt.semilogx(cs, np.maximum(1-1/cs, 0), 'k--', label=r'$\alpha=\max(0,1-1/c)$')
    plt.legend()
    plt.axvline(x=1, ls=':', c='k', lw=0.5)
    plt.axvline(x=2, ls=':', c='k', lw=0.5)
    plt.axhline(y=0.5, ls=':', c='k', lw=0.5)
    plt.contourf(cs, alphas, maxMagA, levels, norm=cnorm,
                 cmap='Greys', extend='both')
    plt.colorbar()
    plt.contour(cs, alphas, maxMagA, [0, 1], colors=['k', 'k'])
    plt.xlabel(r'$c$')
    plt.ylabel(r'$\alpha$')
    saveFig("magA_c_alpha_chi1k"+str(k+1))


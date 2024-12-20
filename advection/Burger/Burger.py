# Outer code for setting up the non-linear advection problem (Burger's eqn)
# on a uniform grid, calling the function to perform the advection and plot.

from advectionSchemes import *
from initialConditions import *

def advection(nx, nt, c, d, plotFreq, plotDir):
    "Advection initial conditions on a"
    "domain between x = xmin and x = xmax split over nx spatial steps"
    "with c = dt/dx, for nt time steps"
    "Hilary Weller"

    # Parameters
    xmin = 0.
    xmax = 1.
    
    # Derived parameters
    dx = (xmax - xmin)/nx

    # spatial points for plotting and for defining initial conditions
    x = np.arange(xmin, xmax, dx)

    # Initial conditions
    phiOld = cosBell(x, 0, 0.4)

    # initialise all schemes
    phiSL   = phiOld.copy()
    phiMPDATAE = phiOld.copy()
    phiMPDATAI = phiOld.copy()
    plotSchemes(x, [phiSL, phiMPDATAE, phiMPDATAI], 
            ['semi-Lagrangian', 'MPDATA explicit', 'MPDATA implicit'], 0,
            plotDir)

    print('Total mass of each = ', np.sum(phiSL), np.sum(phiMPDATAE),
          np.sum(phiMPDATAI))

    # Time step all schemes
    for it in range(1, int(nt/plotFreq)+1):
        print("Time step", it*plotFreq)
        phiSL = semiLagrangian(phiSL, c, d, plotFreq)
        phiMPDATAE = MPDATA(phiMPDATAE, c, d, plotFreq, explicit=1)
        phiMPDATAI = MPDATA(phiMPDATAI, c, d, plotFreq, explicit=0)
        plotSchemes(x, [phiSL, phiMPDATAE, phiMPDATAI], 
                ['semi-Lagrangian', 'MPDATA explicit', 'MPDATA implicit'],
                it, plotDir)
        print('Total mass of each = ', np.sum(phiSL), np.sum(phiMPDATAE),
              np.sum(phiMPDATAI))

K = 0.01/40/.8
advection(nx = 40, nt = 60,c = 0.8, d = K*.8*40,  plotFreq = 3, plotDir='plots1')
advection(nx = 40, nt = 20,c = 2.4, d = K*2.4*40, plotFreq = 1, plotDir='plots2')
advection(nx =120, nt = 60,c = 2.4, d = K*2.4*120,plotFreq = 3, plotDir='plots3')
advection(nx =120, nt =180,c = 0.8, d = K*.8*120, plotFreq = 9, plotDir='plots4')




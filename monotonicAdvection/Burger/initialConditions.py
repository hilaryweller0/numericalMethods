import numpy as np
import os
import matplotlib.pyplot as plt

# Initial conditions function for advection

def squareWave(x,alpha,beta):
    "A square wave as a function of position, x, which is 1 between alpha"
    "and beta and zero elsewhere. The initialisation simply samples the"
    "square wave to set values at grid points"
    "Hilary Weller"

    return np.where((x>=alpha) & (x<=beta), 1., 0.)

def cosBell(x,alpha,beta):
    "A cosine bell as a function of position, x between alpha and beta. Zero elsewhere"
    "Hilary Weller"

    return np.where((x>=alpha) & (x<=beta),
                    0.5*(1-np.cos(2*np.pi*(x-alpha)/(beta-alpha))),
                    0.)

def plotSchemes(x, data, labels, it, plotDir):
    """Plot all the solutions as a function of x in list of arrays data at time
       step it with labels in list of strings labels"""
    
    font = {'size'   : 14}
    plt.rc('font', **font)
    plt.clf()
    c = ["black", "red", "blue", "cyan", "magenta", "green", "grey"]
    for id in range(len(data)):
        plt.plot(x, data[id], label=labels[id], color=c[id])

    plt.axhline(0, linestyle=':', color='black')
    plt.legend() #(loc='upper left')
    plt.xlabel('$x$')
    plt.xlim([0,1])
    plt.ylim([-.01,1.2])
    plt.tight_layout()
    
    if not os.path.exists(plotDir):
        os.mkdir(plotDir)
    plt.savefig(plotDir+"/allSchemes"+str(it)+".pdf")


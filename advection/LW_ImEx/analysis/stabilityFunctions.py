import numpy as np
import matplotlib.pyplot as plt
import os

if not os.path.exists('plots'):
    os.makedirs('plots')

def saveFig(name):
    fileName='plots/'+name+'.pdf'
    print(fileName)
    plt.savefig(fileName)
    os.system("pdfCrop " + fileName + " > /dev/null 2>&1" )
    os.system("evince " + fileName + " > /dev/null 2>&1 &")
    plt.clf()

# Candidate schemes for alpha as a function of c and chi as a function of alpha
def alphaFromc(c):
    return np.maximum(0, 1-1/c)


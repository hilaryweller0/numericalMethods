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

def cosBell(x, a=0, b=0.5):
    return np.where((x>a) & (x<b), np.sin((x-a)/(b-a)*np.pi)**2, 0.)

def squareWave(x, a=0, b=0.5):
    return np.where((x>a) & (x<b), 1., 0.)

def cosine(x):
    return np.cos(2*np.pi*x)

def alphaFromc(c):
    return np.where(c<1, 0, 1-1/c)

def alphaFromcWB(c):
    #d = 0.5
    return np.where(c<2, 0, 0.5*(c-1.9)/c)

def chiFromc(c,k):
    if k == 0:
        return 0
    else:
        return 1 if c <= 2 else (2*c-1)/(3*c-3)

def chiFromcd(c,k,options):
    d = options["damp"]
    if k == 0:
        return 0
    else:
        return 1 if c <= 2 else (2*c-1)*(d*(2*c-1)+1)/(6*c*(c-1))


# Initial condition functions
import numpy as np

def combi(x):
    "Initial conditions consisting of a square wave and a bell"
    return np.where((x > 0) & (x < 0.5), 0.5*(1-np.cos(2*np.pi*x/.5)),
                    np.where((x > 0.6) & (x < 0.8), 1., 0.))

def square(x):
    "Initial conditions consisting of a square wave"
    return np.where((x > 0) & (x < 0.4), 1., 0.)

def cosBell(x):
    "Initial conditions consisting of a  bell"
    return np.where((x > 0) & (x < 0.4), 0.5*(1-np.cos(2*np.pi*x/.4)), 0)

def halfWave(x):
    "Initial conditions consisting of a square wave, rounded off at the back"
    return 0.5*np.where((x > 0) & (x <= 0.2), 0.5*(1-np.cos(np.pi*x/.2)),
                    np.where((x > 0.2) & (x <= 0.4), 1, 0))

def fullWave(x):
    "A positive sine wave"
    return 0.5*(1 - np.sin(2*np.pi*x))

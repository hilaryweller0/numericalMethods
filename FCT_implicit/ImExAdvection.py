import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

# Advection schemes

def PPM(phi, c):
    """Advect profile phi in a periodc domain for one time step with Courant
    number c with PPM"""
    # phi interpolated onto interfaces (i+1/2)
    phiI = 1/12*(-np.roll(phi, 1) + 7*phi + 7*np.roll(phi, -1) - np.roll(phi, -2))

    # Flux at i+1/2
    phiH = (1 - 2*c + c**2)*phiI \
         + (3*c - 2*c**2)*phi \
         + (-c + c**2)*np.roll(phiI,1)
    
    return phi - c*(phiH - np.roll(phiH,1))


def LW_ImEx(phi, c):
    """Advect profile phi in a periodc domain for one time step with Courant
    number c with a trapezoidal implicit, centred in space scheme"""
    
    nx = len(phi)
    phiOld = phi.copy()
    
    # Off centering
    a = max(0, 1-1/max(1e-16, abs(c)))
    chi = max(0, 1-2*a)
    
    # The matrix for the implicit part
    M = np.zeros([nx,nx])
    for i in range(nx):
        M[i,i] = 1 + a*abs(c)
        M[i,(i-1)%nx] = min(-a*c, 0)
        M[i,(i+1)%nx] = min(a*c, 0)
    
    # The ante-diffusion
    phi = phiOld - 0.5*(abs(c)-chi*c**2)*(np.roll(phi,-1) - 2*phi + np.roll(phi,1))
    
    # The explicit bit
    if c >= 0:
        phi += -(1-a)*c*(phiOld - np.roll(phiOld,1))
    else:
        phi += -(1-a)*c*(np.roll(phiOld,-1) - phiOld)

    return la.solve(M, phi)

# Initial condition functions
def combi(x):
    "Initial conditions consisting of a square wave and a bell"
    return np.where((x > 0) & (x < 0.4), 0.5*(1-np.cos(2*np.pi*x/.4)),
                    np.where((x > 0.5) & (x < 0.7), 1., 0.))

def square(x):
    "Initial conditions consisting of a square wave"
    return np.where((x > 0) & (x < 0.4), 1., 0.)

def cosBell(x):
    "Initial conditions consisting of a  bell"
    return np.where((x > 0) & (x < 0.4), 0.5*(1-np.cos(2*np.pi*x/.4)), 0)

def initial(x):
    return combi(x)


# Set up domain and plot time steps
nx = 40
dx = 1/nx
x = np.arange(0, 1, dx)
phi = initial(x)

# For PPM
nt = 50
c = nx/nt
plotFreq = 50
plt.plot(x, phi, label = 't=0')
plt.title('PPM, Courant number = '+str(c))
for it in range(10*nt):
    phi = PPM(phi, c)
    if (it+1)%plotFreq == 0:
        plt.plot(x, phi, label = 'n='+str(it+1))
plt.legend()
plt.savefig('plots/PPM.pdf')
plt.clf()

# For LW_ImEx, c<1 
phi = initial(x)
nt = 50
c = nx/nt
plotFreq = 10
plt.plot(x, phi, label = 't=0')
plt.title('LW_ImEx, Courant number = '+str(c))
for it in range(nt):
    phi = LW_ImEx(phi, c)
    if (it+1)%plotFreq == 0:
        plt.plot(x, phi, label = 'n='+str(it+1))
plt.legend()
plt.savefig('plots/LW_ImEx_c'+str(c)+'.pdf')
plt.clf()

# For LW_ImEx, c<-1
phi = initial(x)
nt = 25
c = -nx/nt
plotFreq = 5
plt.plot(x, phi, label = 't=0')
plt.title('LW_ImEx, Courant number = '+str(c))
for it in range(nt):
    phi = LW_ImEx(phi, c)
    if (it+1)%plotFreq == 0:
        plt.plot(x, phi, label = 'n='+str(it+1))
plt.legend()
plt.savefig('plots/LW_ImEx_c'+str(c)+'.pdf')
plt.clf()

# PPM blended with LW_ImEx
phi = initial(x)
nt = 25
c = nx/nt
cE = min(1.,c)
cI = c-cE
plotFreq = 5
plt.plot(x, phi, label = 't=0')
plt.title('LW_ImEx, Courant number = '+str(c))
for it in range(nt):
    phi =PPM(phi, cE)
    phi = LW_ImEx(phi, cI)
    if (it+1)%plotFreq == 0:
        plt.plot(x, phi, label = 'n='+str(it+1))
plt.legend()
plt.savefig('plots/PPM_LW_ImEx_c'+str(c)+'.pdf')
plt.clf()


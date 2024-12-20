import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

nx = 40
nt = 16
c = 2
dx = 1/nx
x = np.arange(0, 1, dx)

# Off centering
alpha = max(0.5, 1-1/max(c, 1e-12))

# Initial conditions
phi = np.where(x<0.5, 1., 0.)
plt.plot(x, phi)
plt.draw()

# Matrix for implicit part
M = np.zeros([nx,nx])
for i in range(nx):
    M[i,i] = 1 + alpha*c
    M[i,i-1] = -alpha*c

# Time Stepping
HOC = 1/8*(3*np.roll(phi,-1) - 5*phi + np.roll(phi,1) + np.roll(phi,2))
for it in range(nt):
    phiOld = phi - (1-alpha)*c*(phi - np.roll(phi,1)) - (1-alpha)*c*HOC
    # Two iterations
    for itt in range(2):
        HOC = 1/8*(3*np.roll(phi,-1) - 5*phi + np.roll(phi,1) + np.roll(phi,2))
        phi = la.solve(M, phiOld - alpha*c*HOC)
    if (it+1)%1 == 0:
        plt.plot(x, phi)
        plt.draw()
        plt.pause(0.1)
plt.show()

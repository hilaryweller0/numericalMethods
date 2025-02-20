# Test Symmetric Gauss-Seidel for solving the implicit upwind method
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve
from initialConditions import *

# Advection on a one-dimensional, uniform, periodic grid
# Interface j-1/2 is indexed j so that cell j is between interfaces indexed j and j+1
#         |  cell |   
#   j-1   |   j   |   j+1
#         j      j+1

nx = 20
dx = 1/nx

# The locations of the cell centres and the interfaces
x = np.arange(0.5*dx, 1, dx)
xi = np.arange(0, 1, dx)

# The Courant number is defined at interfaces (j+1/2) and varies in space
c = 16*np.cos(2*np.pi*x*nx/15 + 0.5*dx)

# Matrix for the exact solution
M = np.zeros([nx,nx])
for j in range(nx):
    M[j,j] = 1 + max(c[(j+1)%nx],0) + max(-c[j],0)
    M[j,(j-1)%nx] = -max(c[j],0)
    M[j,(j+1)%nx] = -max(-c[(j+1)%nx],0)

# The dependent variable, phi, is defined at cell centres
phi = fullWave(x*4) + 2
phiNew = phi.copy()
phiNewExact = solve(M,phi)

# Forward iteration of upwind advection, backward in time
for j in range(len(phi)):
    phiNew[j%nx] = (
        phi[j%nx] + max(c[j%nx],0)*phiNew[(j-1)%nx]
               + max(-c[(j+1)%nx],0)*phiNew[(j+1)%nx]
               )/(1 + max(c[(j+1)%nx],0) + max(-c[j%nx],0))

# Calculate the residual
R = phiNew.copy()
for j in range(len(phi)):
    R[j] -= (
        phi[j] + max(c[j],0)*phiNew[(j-1)%nx]
               + max(-c[(j+1)%nx],0)*phiNew[(j+1)%nx]
               )/(1 + max(c[(j+1)%nx],0) + max(-c[j],0))
print('First residual is ', R)
print('First error is ', phiNew - phiNewExact)
# Plot the solution after one iteration
plt.plot(x, phi, label='t=0')
plt.plot(x, phiNew, label='One iteration')
plt.plot(x, phiNewExact, label='Full matrix solve')
plt.plot(xi, c, label='Courant number')
plt.legend()
plt.show()

# Backward iteration of upwind advection, backward in time
for j in range(len(phi)-1,-1,-1):
    phiNew[j%nx] = (
        phi[j%nx] + max(c[j%nx],0)*phiNew[(j-1)%nx]
               + max(-c[(j+1)%nx],0)*phiNew[(j+1)%nx]
               )/(1 + max(c[(j+1)%nx],0) + max(-c[j%nx],0))

# Calculate the residual
R = phiNew.copy()
for j in range(len(phi)):
    R[j] -= (
        phi[j] + max(c[j],0)*phiNew[(j-1)%nx]
               + max(-c[(j+1)%nx],0)*phiNew[(j+1)%nx]
               )/(1 + max(c[(j+1)%nx],0) + max(-c[j],0))
print('Second residual is ', R)
print('Second error is ', phiNew - phiNewExact)
# Plot the solution after backward iteration
plt.plot(x, phi, label='t=0')
plt.plot(x, phiNew, label='Forward-backward iterations')
plt.plot(x, phiNewExact, label='Full matrix solve')
plt.plot(xi, c, label='Courant number')
plt.legend()
plt.show()

# Calculate the residual of the exact
R = phiNewExact.copy()
for j in range(len(phi)):
    R[j] -= (
        phi[j] + max(c[j],0)*phiNewExact[(j-1)%nx]
               + max(-c[(j+1)%nx],0)*phiNewExact[(j+1)%nx]
               )/(1 + max(c[(j+1)%nx],0) + max(-c[j],0))
print('Exact residual is ', R)

# Forward again
for j in range(len(phi)):
    phiNew[j%nx] = (
        phi[j%nx] + max(c[j%nx],0)*phiNew[(j-1)%nx]
               + max(-c[(j+1)%nx],0)*phiNew[(j+1)%nx]
               )/(1 + max(c[(j+1)%nx],0) + max(-c[j%nx],0))
# Plot the solution after second forward-iteration
plt.plot(x, phi, label='t=0')
plt.plot(x, phiNew, label='Forward-backward-forward iterations')
plt.plot(x, phiNewExact, label='Full matrix solve')
plt.plot(xi, c, label='Courant number')
plt.legend()
plt.show()

# Calculate the residual
R = phiNew.copy()
for j in range(len(phi)):
    R[j] -= (
        phi[j] + max(c[j],0)*phiNew[(j-1)%nx]
               + max(-c[(j+1)%nx],0)*phiNew[(j+1)%nx]
               )/(1 + max(c[(j+1)%nx],0) + max(-c[j],0))
print('Third residual is ', R)


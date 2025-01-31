"""Backward Euler in 2D using sparse matrices"""
"""Hilary Weller"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.linalg as sla
import scipy.sparse.linalg
from scipy import sparse
from time import process_time
from scipy.sparse import dok_matrix
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import spilu 

# Array sizes
nx = 101
ny = 101
nt = 500

# Grid spacing
dx = 1/nx
dy = 1/ny
dt = 1/nt

# Domain of cell centres and edges
xc = np.linspace(0.5*dx, 1, nx)
yc = np.linspace(0.5*dy,1,ny)
xe = np.linspace(0,1,nx+1)
ye = np.linspace(0,1,ny+1)

# Initial conditions (in 2D matrix form) (with ghost cells for boundaries)
phi = np.zeros([nx+2, ny+2])
centerInit = [0.5, 0.75]
widthInit = 0.12
for i in range(nx):
    for j in range(ny):
        dist = np.sqrt((xc[i] - centerInit[0])**2 + (yc[j] - centerInit[1])**2)
        if dist < widthInit:
            phi[i+1,j+1] = .25*(1 + np.cos(np.pi*dist/widthInit))**2

# Plot initial conditions
plt.contourf(xc,yc,phi[1:-1,1:-1].transpose())
plt.draw()

# Streamfunction and constant divergence free velocity field
streamFunc = np.zeros([nx+1,ny+1])
rotationCenter = [0.5,0.5]
rotationRate = -np.pi
for i in range(nx+1):
    for j in range(ny+1):
        distSqr = (xe[i] - rotationCenter[0])**2 \
                + (ye[j] - rotationCenter[1])**2
        streamFunc[i,j] = distSqr*rotationRate

# Velocity field from stream function (in matrix form, on edges)
u = np.zeros([nx+1, ny])
v = np.zeros([nx, ny+1])
for i in range(nx-1):
    for j in range(ny):
        u[i+1,j] = (streamFunc[i+1,j+1] - streamFunc[i+1,j])/dy

for i in range(nx):
    for j in range(ny-1):
        v[i,j+1] = -(streamFunc[i+1,j+1] - streamFunc[i,j+1])/dx

# Find maximum Courant number
cMax = np.max([np.max(u)*dt/dx,np.max(v)*dt/dy])
print('Maximum Courant number = ', cMax)

# Create a sparse matrix to do the forward Euler advection
N = (nx+2)*(ny+2)
M_FE = dia_matrix((N,N))
# Diagonal elements of the matrix
diag = np.zeros(N)
diag[0] = 1
diag[-1] = 1
for i in range(1,nx+1):
    for j in range(1,ny+1):
        #Index into diag
        ij = j*(nx+1)+i
        diag[ij] = 1 - 0.5*dt/dx*\
            ((u[i,j-1] + abs(u[i,j-1])) - (u[i-1,j-1] - abs(u[i-1,j-1]))) \
            - 0.5*dt/dy*\
            ((v[i-1,j] + abs(v[i-1,j])) - (v[i-1,j-1] - abs(v[i-1,j-1])))

M_FE.setdiag(diag, k=0)

# 2D advection (explicit)
for it in range(nt):
    phiOld = phi.copy()
    for i in range(1,nx+1):
        for j in range(1,ny+1):
            phi[i,j] = phiOld[i,j] - 0.5*dt/dx*\
            (\
                (u[i,j-1] + abs(u[i,j-1]))*phiOld[i,j]
              + (u[i,j-1] - abs(u[i,j-1]))*phiOld[i+1,j]
              - (u[i-1,j-1] + abs(u[i-1,j-1]))*phiOld[i-1,j]
              - (u[i-1,j-1] - abs(u[i-1,j-1]))*phiOld[i,j]
            )\
            - 0.5*dt/dy*\
            (\
                (v[i-1,j] + abs(v[i-1,j]))*phiOld[i,j]
              + (v[i-1,j] - abs(v[i-1,j]))*phiOld[i,j+1]
              - (v[i-1,j-1] + abs(v[i-1,j-1]))*phiOld[i,j-1]
              - (v[i-1,j-1] - abs(v[i-1,j-1]))*phiOld[i,j]
            )
    # Plot each time step
    plt.clf()
    plt.contourf(xc,yc,phi[1:-1,1:-1].transpose())
    plt.draw()
    plt.pause(0.001)
plt.show()

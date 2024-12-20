#!/usr/bin/env python3
# -*- coding: utf-8 -*-




## this script runs three numerical methods, all are implicit in First half and explict in the second half of the domain, and use simple 1st order in time. 

## Dirichlet bc of zero. 


"""
@author: jmw418
"""
###Packages###
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.special
from numpy import linalg as LA
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from time import process_time 
from matplotlib import animation
from scipy import interpolate
import matplotlib.pyplot as plt

# initial condition
def IC(x):
    square = np.zeros_like(x)
    square[np.where(x <= 0.1)] = 1.0
    
    chair = np.zeros_like(x)
    for i in range(0,len(x)):
        if (x[i]<0.9):
            if (x[i]>0.8):
                chair[i] = 1
    
    return chair
   
def main():
    ###***    initialisation of constants  ***###
    nx = 200;nt = 200;a = 0.94;tstart = 0;tend = 0.1;xmin=0;xmax = 1;
    
    ###••••• derived parameters •••••###
    dx = (xmax-xmin)/(nx); dt = (tend-tstart)/(nt); c = dt/dx*a;
    print("clf number=", c)

    
    ###••••• initialisation of structures •••••###
    x = np.linspace(xmin,xmax,nx); time = np.linspace(tstart,tend,nt)
    Phi = np.zeros([nx]); Phinew = np.zeros([nx])
    A = np.zeros([nx,nx]); beta = np.zeros([nx])
    
    ###***   set the initial condition ***###
    Phi = IC(x)
    tv = np.zeros(nt)
    err = np.zeros(nt)
    
    ###***   Loop for space ***###
    
    t1_start = process_time() 
    for i in range(0,nt):
    
        
    # Propagate one time step by either Half, or Half22
        Phinew = both(nx,c,A,beta,Phi,Phinew) 
        Phi = Phinew ### restart the Phi
        
        
        tv[i] = Total_variation(Phi)
        Animate(Phi,x,i,a,dt,tv)
        if i<nt-1:
            if tv[i+1]>tv[i]:
                print("TVD violation")
        err[i] = np.linalg.norm(IC((x-i*a*dt)%1 ) - Phi,1)
        
    
    plt.show()
    plt.title("Total Variation with time")
    plt.plot(tv)
    plt.show()
    plt.title("l1 error with time")
    plt.plot(err)
    plt.show()
    
    
    
    
    

###                          ••••• Plotting tools •••••                            

    
def Animate(Phi,x,i,a,dt,tv) :
    plt.title(tv[i])
    plt.axis([0, 1, -0.1, 1.3])
    plt.plot(x,Phi,'b')
    plt.plot(x,IC((x-i*a*dt)%1 ),'r')
    plt.draw()
    plt.pause(0.003)
    plt.clf()
    return
def Total_variation(Phi):
    nx = len(Phi);
    tv = 0
    for j in range(0,nx):
        tv += abs( Phi[j] - Phi[(j-1)%nx] )
    return tv
    

    

###       ••••• Finite Volume Schemes •••••                                    
    
def upstream_flux(m,p,c):
    return ( p*neg(c) + m*pos(c) )    
def pos(p):
    return 0.5*(p+abs(p))
def neg(m):
    return 0.5*(m-abs(m))
    
    
def both(nx,c,A,beta,Phi,Phinew):
    A = np.zeros([nx,nx]);
    do = np.zeros(nx);
    
    inde = np.zeros(nx);indi = np.zeros(nx);
    for i in range(0,nx):
        m = round(nx/2)
        if i > m:
            inde[i] = 1
            indi[i] = 0
        else:
            inde[i] = 0
            indi[i] = 1
            
    for j in range(0,nx):
        do[j] = upstream_flux(Phi[(j-1)%nx],Phi[(j)%nx],c) # this is face j-0.5
    
    for j in range(0,nx):
        Phinew[j] = Phi[j] - inde[j]*( do[(j+1)%nx] - do[j%nx] )
        
    for j in range(0,nx):
        A[j,j] = 1  + indi[j]*(pos(c) - neg(c)) ;
        A[j,(j+1)%nx] = +neg(c)*indi[j];
        A[j,(j-1)%nx] = -pos(c)*indi[j];
    
    Phinew = np.linalg.solve(A,Phinew)
        
    return Phinew
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
    
    




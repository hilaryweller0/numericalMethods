#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# this code contains code for naive implementations of blended schemes. 

# we multiply by indicator functions and use different timestepping schemes. 

## one question is the order in which one does these and how it effects the boundary between them


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


# ------------------------------------------------------------------------------------
# initial condition
def IC(x):
    chair = np.zeros_like(x)
    for i in range(0,len(x)):
        if (x[i]<0.95):
            if (x[i]>0.45):
                chair[i] = 1
    return chair ##+ np.cosh(10*(x-0.25))**(-2) 
    
# ------------------------------------------------------------------------------------
###                       ••••• Main function for 1 step methods •••••            ###
# ------------------------------------------------------------------------------------

def main():
    ###***    initialisation of constants  ***###
    nx = 300;nt = 100;a = 1.9;tstart = 0;tend = 1.0;xmin=0;xmax = 1;
    
    ###••••• derived parameters •••••###
    dx = (xmax-xmin)/(nx); dt = (tend-tstart)/(nt); c = dt/dx*a;
    print("clf number=", c)

    
    ###••••• initialisation of structures •••••###
    x = np.linspace(xmin,xmax,nx); time = np.linspace(tstart,tend,nt)
    Phi = np.zeros([nx]); Phinew = np.zeros([nx])
    A = np.zeros([nx,nx]); beta = np.zeros([nx])
    
    ###***   set the initial condition ***###
    Phi = IC(x)
    EXP = indEXP(x)
    IMP = indIMP(x)
    tv = np.zeros(nt)
    err = np.zeros(nt)
    ###***   Loop for space ***###
    
    t1_start = process_time() 
    for i in range(0,nt):
        
        ###••••• Choose the scheme •••••###
        ###••••• Choose the scheme •••••###
        Phinew = indBESSP33(nx,c,Phi,IMP,EXP) 
        ###••••• Choose the scheme •••••###
        ###••••• Choose the scheme •••••###
        Phi = Phinew ### restart the Phi
        tv[i] = Total_variation(Phi)
        
        Animate(Phi,x,i,a,dt,tv)
        if i>1:
            if (tv[i]>tv[i-1]):
                print("tv violation")
    
    plt.show()
    plt.title("Total Variation with time")
    plt.plot(tv)
    plt.show()
    
    
    
# ------------------------------------------------------------------------------------
###                          ••••• Plotting tools •••••                            ###
# ------------------------------------------------------------------------------------

    
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
    
    

# ------------------------------------------------------------------------------------
###                          ••••• Schemes •••••                                   ###
# ------------------------------------------------------------------------------------

    
    
def indEXP(x):
    EXP = np.zeros_like(x)
    for i in range(0,len(x)):
        if (x[i]>0.5):
                EXP[i] = 1
    return EXP
def indIMP(x):
    IMP = np.zeros_like(x)
    for i in range(0,len(x)):
        if (x[i]<=0.5):
            
                IMP[i] = 1
    return IMP

    
def indBEFE(nx,c,Phi,IMP,EXP):
    
    ## forward Euler on the right hand side
    
    
    Phiprime = np.zeros(nx)
    Phiprime[:] = Phi[:] - c*EXP*( Phi[:] - np.roll(Phi[:],1) )
    
    
    ## implicit Euler on the left hand side
    B = np.zeros([nx,nx])
    for j in range(0,round(nx/2)):
        B[j,j] = 1  + c;
    for j in range(0,round(nx/2)):
        B[j,(j-1)%nx] = -c;
    for j in range(round(nx/2),nx):
        B[j,j] = 1
    
    
    
    return np.linalg.solve(B,Phiprime)
    

def indBESSP33(nx,c,Phi,IMP,EXP):
    
    B = np.zeros([nx,nx])
 
    for j in range(0,round(nx/2)):
        B[j,j] = 1  + c;
    for j in range(0,round(nx/2)):
        B[j,(j-1)%nx] = -c;
    for j in range(round(nx/2),nx):
        B[j,j] = 1
    
    Phiprime = np.zeros(nx)
    
    Phi1 = np.zeros([nx]);Phi2 = np.zeros([nx]);
        
    ##explicit part
    
    Phi1[:] = Phi[:] - c*EXP*( Phi[:] - np.roll(Phi[:],1) )
    Phi2[:] = 3/4*Phi + 1/4*Phi1[:] - 1/4*c*EXP*( Phi1[:] - np.roll(Phi1[:],1) )
    Phiprime[:] = 1/3*Phi[:] + 2/3*Phi2[:] - 2/3*c*EXP*(Phi[:] - np.roll(Phi[:],1))
        
    return np.linalg.solve(B,Phiprime)
    

def indCNSSP33(nx,c,Phi,IMP,EXP):
        
    ##explicit part
    Phi1 = np.zeros([nx]);Phi2 = np.zeros([nx]);Phiprime = np.zeros(nx);
    Phi1[:] = Phi[:] - c*EXP*( Phi[:] - np.roll(Phi[:],1) )
    Phi2[:] = 3/4*Phi + 1/4*Phi1[:] - 1/4*c*EXP*( Phi1[:] - np.roll(Phi1[:],1) )
    Phiprime[:] = 1/3*Phi[:] + 2/3*Phi2[:] - 2/3*c*EXP*(Phi[:] - np.roll(Phi[:],1))
    
    
    ## implicit part
    
    B = np.zeros([nx,nx])
 
    for j in range(0,round(nx/2)):
        B[j,j] = 1  + 0.5*c;
    for j in range(0,round(nx/2)):
        B[j,(j-1)%nx] = -0.5*c;
    for j in range(round(nx/2),nx):
        B[j,j] = 1
    
    Phiprime[:] = Phiprime[:] - 0.5*c*IMP*(Phiprime[:] - np.roll(Phiprime[:],1))
    
        
    
        
    return np.linalg.solve(B,Phiprime)
    
    
    

def indCNSSP104(nx,c,Phi,IMP,EXP):
        
    ## this is an interesting scheme as the implicit scheme has radius of monotonicity less than the explicit method so we get tvd violation in the implicit part.    
        
        
    ##explicit part
    Phi1 = np.zeros([nx]);Phi2 = np.zeros([nx]);Phiprime = np.zeros(nx);
    Phi1[:] = Phi[:]
    for k in range(0,4):
        Phi1[:] = Phi1[:] - 1/6*c*EXP*( Phi1[:] - np.roll(Phi1[:],1) )
    Phi2 = 3/5*Phi[:] +2/5*Phi1[:] - 1/15*c*EXP*( Phi1[:] - np.roll(Phi1[:],1) )
    for k in range(0,4):
        Phi2[:] = Phi2[:]  - 1/6*c*EXP*( Phi2[:] - np.roll(Phi2[:],1) )
    Phiprime[:] = 1/25*Phi[:] + 9/25*Phi1[:] +3/5*Phi2[:]  \
     -3/50*c*EXP*( Phi1[:] - np.roll(Phi1[:],1) )          \
     -1/10*c*EXP*( Phi2[:] - np.roll(Phi2[:],1) )    
    
    ## implicit part
    
    B = np.zeros([nx,nx])
 
    for j in range(0,round(nx/2)):
        B[j,j] = 1  + 0.5*c;
    for j in range(0,round(nx/2)):
        B[j,(j-1)%nx] = -0.5*c;
    for j in range(round(nx/2),nx):
        B[j,j] = 1
    
    Phiprime[:] = Phiprime[:] - 0.5*c*IMP*(Phiprime[:] - np.roll(Phiprime[:],1))
    
        
    
        
    return np.linalg.solve(B,Phiprime)
    
    

def indCN3sSSP104(nx,c,Phi,IMP,EXP):
    ## here we show that ssp104 is orders of magnitude faster than crank nicholson
    ## ceff = 3/5 > 0.5 heun. so is better than Heun. 
    ## these low storage ssp runge kutta tend to the ceff of forward euler, but can attain more properties like accuracy on the way. 
    ## There seems to be no motivation of using implicit schemes after doing this other than unconditional CFL properties. 
    
    
    exp_start = process_time()
    
    ## this is a scheme designed to be ssp for cfl 6, 
        
        
    ##explicit part
    Phi1 = np.zeros([nx]);Phi2 = np.zeros([nx]);Phiprime = np.zeros(nx);
    Phi1[:] = Phi[:]
    for k in range(0,4):
        Phi1[:] = Phi1[:] - 1/6*c*EXP*( Phi1[:] - np.roll(Phi1[:],1) )
    Phi2 = 3/5*Phi[:] +2/5*Phi1[:] - 1/15*c*EXP*( Phi1[:] - np.roll(Phi1[:],1) )
    for k in range(0,4):
        Phi2[:] = Phi2[:]  - 1/6*c*EXP*( Phi2[:] - np.roll(Phi2[:],1) )
    Phiprime[:] = 1/25*Phi[:] + 9/25*Phi1[:] +3/5*Phi2[:]  \
     -3/50*c*EXP*( Phi1[:] - np.roll(Phi1[:],1) )          \
     -1/10*c*EXP*( Phi2[:] - np.roll(Phi2[:],1) )    
    exp_stop = process_time()
    imp_start = process_time() 
    ## implicit part
    
    s = 3
    B = np.zeros([nx,nx])
    for j in range(0,round(nx/2)):
        B[j,j] = 1  + 0.5*c/s;
    for j in range(0,round(nx/2)):
        B[j,(j-1)%nx] = -0.5*c/s;
    for j in range(round(nx/2),nx):
        B[j,j] = 1
    for i in range(0,s):
    
        Phiprime[:] = Phiprime[:] - 0.5*c/s*IMP*(Phiprime[:] - np.roll(Phiprime[:],1))
    
        Phiprime = np.linalg.solve(B,Phiprime)
    imp_stop = process_time() 
        
    print("Explicit time:", -exp_start+exp_stop) 
    print("implicit time:", -imp_start+ imp_stop) 
    return Phiprime
    

def SSP104(nx,c,Phi,IMP,EXP):
    ## this is a scheme designed to be ssp for cfl 6, 
    ##explicit part
    Phi1 = np.zeros([nx]);Phi2 = np.zeros([nx]);Phiprime = np.zeros(nx);
    Phi1[:] = Phi[:]
    for k in range(0,4):
        Phi1[:] = Phi1[:] - 1/6*c*( Phi1[:] - np.roll(Phi1[:],1) )
    Phi2 = 3/5*Phi[:] +2/5*Phi1[:] - 1/15*c*( Phi1[:] - np.roll(Phi1[:],1) )
    for k in range(0,4):
        Phi2[:] = Phi2[:]  - 1/6*c*( Phi2[:] - np.roll(Phi2[:],1) )
    Phiprime[:] = 1/25*Phi[:] + 9/25*Phi1[:] +3/5*Phi2[:]  \
     -3/50*c*( Phi1[:] - np.roll(Phi1[:],1) )          \
     -1/10*c*( Phi2[:] - np.roll(Phi2[:],1) )    
    return Phiprime
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
    
    




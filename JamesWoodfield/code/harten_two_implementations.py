#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        if (x[i]<0.7):
            if (x[i]>0.2):
                chair[i] = 1
    return chair 
    
# ------------------------------------------------------------------------------------
###                       ••••• Main function for 1 step methods •••••            ###
# ------------------------------------------------------------------------------------

def main():
    ###***    initialisation of constants  ***###
    nx = 100;nt = 100;a = 0.45;tstart = 0;tend = 2.0;xmin=0;xmax = 1;
    
    ###••••• derived parameters •••••###
    dx = (xmax-xmin)/(nx); dt = (tend-tstart)/(nt); c = dt/dx*a;
    print("clf number=", c)

    
    ###••••• initialisation of structures •••••###
    x = np.linspace(xmin,xmax,nx); time = np.linspace(tstart,tend,nt)
    Phi = np.zeros([nx]); Phinew = np.zeros([nx])
    Phiff = np.zeros([nx]);Phiffnew = np.zeros([nx]);
    
    ###***   set the initial condition ***###
    Phi = IC(x); Phiff= IC(x);
    aa, bb = ab(x); EXP = indEXP(x); IMP = indIMP(x);
    tv = np.zeros(nt)
    err = np.zeros(nt)
    ###***   Loop for space ***###
    
    for i in range(0,nt):

        ###••••• Choose the scheme •••••###
        ###••••• Scheme One •••••###
        Phiffnew = indBESSP1(nx,c,Phiff,IMP,EXP,aa,bb) 
        ###••••• Scheme two •••••###
        Phinew = indBEFE(nx,c,Phi,IMP,EXP,aa,bb) 
        ###••••• Choose the scheme •••••###
        Phi = Phinew 
        Phiff = Phiffnew
        print(np.linalg.norm(Phi-Phiff))
        #tv[i] = Total_variation(Phi)
        
        Animate(Phinew,Phiffnew,x,i,a,dt,tv)
        if i>1:
            if (tv[i]>tv[i-1]+0.00000001):
                print("tv violation")
    
    plt.show()
    #plt.title("Total Variation with time")
    #plt.plot(tv)
    plt.show()
    
    
    
# ------------------------------------------------------------------------------------
###                          ••••• Plotting tools •••••                            ###
# ------------------------------------------------------------------------------------

    
def Animate(Phi,Phiff,x,i,a,dt,tv) :

    plt.clf()
    plt.title(tv[i])
    plt.axis([0, 1, -0.1, 1.3])
    plt.plot(x,Phi,'r',linewidth = 1.01)
    plt.plot(x,Phiff,'b',linewidth = 1.01)
    plt.plot(x,IC((x-i*a*dt)%1 ),'k',linewidth = 1.01)
    plt.draw()
    plt.pause(0.003)
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

    
def ab(x):
    ## calcs the min and max implicit points 
    nx = len(x)
    vect = np.zeros(nx)
    for i in range(0,len(x)):
        if (x[(i)%nx]<=0.75):
            if (x[(i)%nx]>=0.25):
                vect[i] = i
                print(i)
    aa = int(np.min(vect[np.nonzero(vect)]))
    bb = int(np.max(vect[np.nonzero(vect)]))
    print(np.min(vect[np.nonzero(vect)]))
    print(np.max(vect[np.nonzero(vect)]))
    return aa, bb
    
def indEXP(x):
    EXP = np.zeros_like(x)
    for i in range(0,len(x)):
        if (x[i]>=0.75):
            EXP[i] = 1
        if (x[i]<=0.25):
            EXP[i] = 1
    plt.plot(EXP)
    print(EXP)
    return EXP
    
def indIMP(x):
    IMP = np.zeros_like(x)
    for i in range(0,len(x)):
        if (x[i]<0.75):
            if (x[i]>0.25):
                IMP[i] = 1
    plt.plot(IMP)
    print(IMP)
    plt.show()
    return IMP
    
    
def indBEFE(nx,c,Phi,IMP,EXP,aa,bb):
    ## this implementation doesn't use the explicit ##
    Phiprime = np.zeros(nx)
    cj = -min(0,c)## represents cj+0.5
    dj = max(c,0)## represents dj-0.5
    Phiprime[:] = Phi[:] + cj*EXP*(np.roll(Phi[:],-1) - Phi[:]) - dj*EXP*(Phi[:] - np.roll(Phi[:],1)) 
    
    B = np.identity(nx)
    for j in range(0,nx):
        B[j,(j-1)%nx] = -IMP[j]*dj;
        B[j,j]        = 1 + IMP[j]*cj + IMP[j]*dj;
        B[j,(j+1)%nx] = -IMP[j]*cj;
        
    return np.linalg.solve(B,Phiprime)
    
    
def indBESSP1(nx,c,Phi,IMP,EXP,aa,bb):
    ## this implementation uses the explicit method ##
    Phiprime = np.zeros(nx)
    cj = -min(c,0)## represents cj+0.5 ## this is zero
    dj = max(c,0)## represents dj-0.5
    Phiprime[:] = Phi[:] + cj*EXP*(np.roll(Phi[:],-1) - Phi[:]) - dj*EXP*(Phi[:] - np.roll(Phi[:],1)) 
    
    B = np.identity(nx)
    for j in range(0,nx):
        B[j,(j-1)%nx] =   - IMP[j]*dj;
        B[j,j]        = 1 + IMP[j]*cj + IMP[j]*dj;
        B[j,(j+1)%nx] =   - IMP[j]*cj;
        
    ## The below implements the explicit 
    Phiprime[aa] = Phi[aa] + dj*(Phiprime[aa-1]) 
    Phiprime[bb] = Phi[bb] + cj*(Phiprime[bb+1]) 
 
    B[aa,aa] = 1+cj+dj
    B[aa,aa-1] = 0
    B[aa,aa+1] = -cj
    
    B[bb,bb]     = 1 + dj +cj
    B[bb,bb-1]   = -dj 
    B[bb,bb+1]   = 0
    
        
    return np.linalg.solve(B,Phiprime)
    
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
    
    




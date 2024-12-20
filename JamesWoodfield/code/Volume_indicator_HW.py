#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jmw418
"""

###Packages###
import numpy as np
import scipy.linalg
from numpy import linalg 
import matplotlib.pyplot as plt

from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
from matplotlib.collections import LineCollection
import imageio
import os
import glob
import sys
import time
from PIL import Image
import natsort 

# ------------------------------------------------------------------------------------
# initial condition
def IC(x):
    chair = np.zeros_like(x)
    for i in range(0,len(x)):
        if ((x[i]>0.2) & (x[i]<0.3)):
            chair[i] = 1
    return chair #+ np.cos(30*(x-0.05))**2*(x<=0.1)*(x>=0.0)
# ------------------------------------------------------------------------------------
###                       ••••• Main function for 1 step methods •••••            ###
# ------------------------------------------------------------------------------------

def main():
    ## resolution of saved images.
    res = 300
    ###***    initialisation of constants  ***###
    nx = 200;nt = 200;a = 0.45;tstart = 0;tend = 1.;xmin=0;xmax = 1;
#    nx = 200;nt = 200;a = 0.95;tstart = 0;tend = 0.5;xmin=0;xmax = 1;
#    nx = 200;nt = 100;a = 0.95;tstart = 0;tend = 0.5;xmin=0;xmax = 1;

    ###••••• derived parameters •••••###
    dx = (xmax-xmin)/(nx); dt = (tend-tstart)/(nt); c = dt/dx*a;
    
    ###••••• initialisation of structures •••••###
    x = np.linspace(xmin,xmax,nx); time = np.linspace(tstart,tend,nt); Phi = np.zeros([nx]); 
    
    ###***   initial condition ***###
    Phi = IC(x); 
    EXP = indEXP(x); IMP = indIMP(x);
    print("initial Condition Set")
    
    ###****  Storing quantities  ****###
    Store = np.zeros([nt,nx])
    tv = np.zeros(nt);err1 = np.zeros(nt);err2 = np.zeros(nt);
    mass = np.zeros(nt);energy = np.zeros(nt);variance = np.zeros(nt);
    
    
    ###***  Numerical Scheme ***###
    scheme = 'fiBEFE'
    
    ###***   Loop for time ***###
    
    print('----------------------------------')
    print( '  Final time : ',tend)
    print('  Time step  : ',dt)
    print('  CFL        : ',c)
    print('  Nb points  : ',nx)
    print('  Scheme  : ',scheme)
    print('----------------------------------')
    
    f= open(''+str(scheme)+'.txt','w+')
    f.write('Scheme  :'+str(scheme)+'\n')
    f.write('Final time :'+str(tend)+'\n')
    f.write('CFL :'+str(c)+'\n')
    f.write('Nb points  :'+str(nx)+'\n')
    print('----------------------------------')
    print("saving initial conditions")
    ICSaver(Phi,x,c,scheme,IMP,EXP,res)
    print("saved initial conditions")
    print('----------------------------------')
    
    print("Starting Scheme")
    for i in range(0,nt):
        tv[i] = Total_variation(Phi)
        mass[i],energy[i],variance[i] = Measures(Phi,x,i,a,dt)
        Animate(Phi,x,i,a,dt,tv)
        #AnimationSaver(Phi,x,i,a,dt,tv,c,scheme)
        Store[i,:] = Phi[:]
        
        
        ###••••• Loop for Space•••••###
        if( scheme=='BEK' ):
            Phi =  BEK(nx,c,Phi)
        if( scheme=='BE' ):
            Phi =  BE(nx,c,Phi)
        if( scheme=='FEK' ):
            Phi =  FEK(nx,c,Phi)
        if( scheme=='FE' ):
            Phi =  FE(nx,c,Phi)
            
        if( scheme=='CNK' ):
            Phi =  CNK(nx,c,Phi)
        if( scheme=='CN' ):
            Phi =  CN(nx,c,Phi)  
        if( scheme=='viBEFE'):    
            Phi = viBEFE(nx,c,Phi,IMP,EXP)
        if( scheme=='viBEFEK'):    
            Phi = viBEFEK(nx,c,Phi,IMP,EXP)
        if( scheme=='fiBEFE'):    
            Phi = fiBEFE(nx,c,Phi,IMP,EXP)
            
        if( scheme=='viBESSP104'):   
            Phi = viBESSP104(nx,c,Phi,IMP,EXP)
        if( scheme=='viBESSP104Kk'):
            Phi = viBESSP104Kk(nx,c,Phi,IMP,EXP)
        if( scheme=='viBESSP104K'):
            Phi = viBESSP104K(nx,c,Phi,IMP,EXP)
            
               
        if( scheme=='fiBESSP104K'):
            Phi =fiBESSP104K(nx,c,Phi,IMP,EXP)
        if( scheme=='fiBESSP104Kk'):
            Phi =fiBESSP104Kk(nx,c,Phi,IMP,EXP)
            
            

        if( scheme=='TRBDF2K' ):
            Phi =  TRBDF2K(nx,c,Phi)
        if( scheme=='TRBDF2' ):
            Phi =  TRBDF2(nx,c,Phi)
        if( scheme=='SDIRK2K' ):
            Phi =  SDIRK2K(nx,c,Phi)
        if( scheme=='SDIRK2' ):
            Phi =  SDIRK2(nx,c,Phi)
            
            
            

        if i>=1:
            if (tv[i]>tv[i-1]+0.00000000001):
                print("tv violation:", abs(tv[i-1]-tv[i]))
                print(tv[i-1],tv[i])
        #print(tv[i])
        
        
        
        
    ### Plot Mod Time ###
        if (i == int((nt-1)/2)):
            print('----------------------------------')
            print("Saving final time")
            plt.clf()
            plt.ylabel('y')
            plt.xlabel('x')
            plt.axis([0, 1, -0.1, 1.3])
            plt.plot(x,Phi,'b',label = 'numerical '+str(scheme)+' sol')
            plt.plot(x,IC((x-i*a*dt)%1 ),'r',label = 'Analytic')
            plt.legend()
            plt.title(''+str(scheme)+'')
            plt.savefig(''+str(scheme)+'-MT.png',dpi = res)
            print("Saved mid time")
            print('----------------------------------')
    
    ### Plot Final Time ###
        if (i == nt-1):
            print('----------------------------------')
            print("Saving final time")
            plt.clf()
            plt.ylabel('y')
            plt.xlabel('x')
            plt.axis([0, 1, -0.1, 1.3])
            plt.plot(x,Phi,'b',label = 'numerical '+str(scheme)+' sol')
            plt.plot(x,IC((x-i*a*dt)%1 ),'r',label = 'Analytic')
            plt.legend()
            plt.title(''+str(scheme)+'')
            plt.savefig(''+str(scheme)+'-FT.png',dpi = res)
            print("Saved final time")
            print('----------------------------------')
    
    ### Plot Total Variation ###
    plt.clf()
    plt.ylabel('y')
    plt.xlabel('time')
    plt.plot(time,tv,linewidth = 0.8,label = 'total variation')
    plt.title('Total Variation '+str(scheme)+'')
    plt.legend()
    plt.savefig(''+str(scheme)+'-TV.png',dpi = res)
    print("Plotted Total Variation")
    print('----------------------------------')
    
    plt.clf()
    plt.ylabel('y')
    plt.xlabel('time')
    plt.plot(time,mass,linewidth = 0.8,label = 'mass')
    plt.plot(time,energy,linewidth = 0.8,label = 'energy')
    plt.title('energy mass against time '+str(scheme)+'')
    plt.legend()
    plt.savefig(''+str(scheme)+'-conservation.png',dpi = res)
    
    
    ### Plot ratio of successive Total variations.
    plt.clf()
    tvr = np.zeros(nt)
    for i in range(1,nt):
        tvr[i]= tv[i]/tv[i-1]
    plt.ylabel('y')
    plt.xlabel('time')
    plt.title(''+str(scheme)+' Total Variation ratio with time')
    plt.plot(time[1:],tvr[1:],linewidth = 0.8,label = 'relative total variation')
    plt.plot(time[1:],1+0*time[1:],'k',linewidth = 0.8,label = '1')
    plt.legend()
    plt.savefig(''+str(scheme)+'-TVR.png',dpi = res)
    
    print("Plotted Total Variation Ratio")
    print('----------------------------------')
    ### plot space time diagram
    Printer(Store,x,time,scheme,c,res)
    print("Plotted space time diagram")
    print('----------------------------------')
    #giffMaker2(scheme)
    
# ------------------------------------------------------------------------------------
###                          ••••• Limiter Used •••••                                   ###
# ------------------------------------------------------------------------------------

def HO(r):
    return 1
def LO(r):
    return 0
    
#### the below have lim r tends to 1. 
def min_mod(r):
    return max(0,min(1,r)) 
def van_Albada(r):
    return (r**2+r)/(r+1)
    
### the below have lim r tending to 2.
def VanL(r):
    return (r+abs(r))/(1+abs(r))
def superbee(r):
    return max(0,min(2*r,1),min(r,2))
    
def Limiter(r):### the limiter is turned off
    return LO(r)

# ------------------------------------------------------------------------------------
###                          ••••• BASE Schemes •••••                ###
# ------------------------------------------------------------------------------------
## we make all higher order schemes out of 2 timestepping schemes, 
## TimeSchemes: Forward Euler, Backward Euler
## Space schemes: Each of the above can be implemented with a Limiter to get higher order in space.
## Note The choice of the spatial scheme can be chosen differently, here is a 3rd order upwind scheme. 
## the choice of spatial scheme is very important. 

def FE(nx,c,Phi):
    return Phi - c* (np.roll(Phi[:],0) - np.roll(Phi[:],1))

def FEK(nx,c,Phi):
    
    ep = 10**(-8)
    r = np.zeros(nx); rr = np.zeros(nx);
    Phihalf = np.zeros(nx)
    r =  ( np.roll(Phi[:],0) - np.roll(Phi[:],1) ) / ( np.roll(Phi[:],-1) - np.roll(Phi[:],0) +ep)
    for j in range(0,nx):
        rr[j] = Limiter(r[j%nx])
    Phihalf[:]  = Phi[:] + 1/2*rr*(  np.roll(Phi[:],-1) - np.roll(Phi[:],0) ) ### second order correction method
    return Phi[:] - c*( np.roll(Phihalf[:],0) - np.roll(Phihalf[:],1) )
    
def BE(nx,c,Phi):
    B = np.identity(nx)
    for j in range(0,nx):
        B[j,j] = 1  + c;
        B[j,(j-1)%nx] = - c ;
    return scipy.linalg.solve(B,Phi)
    
def BEK(nx,c,Phi):
    ep = 10**(-8)
    r = np.zeros(nx)
    for j in range(0,nx):
        r[j] = (Phi[j%nx] - Phi[(j-1)%nx]) / (Phi[(j+1)%nx] - Phi[j%nx] +ep)
    B = np.identity(nx)
    for j in range(0,nx):
        B[j,j] = 1  + c - c/2*Limiter(r[j%nx]) - c/2*Limiter(r[(j-1)%nx]) ;
        B[j,(j+1)%nx] = c/2*Limiter(r[j%nx]);
        B[j,(j-1)%nx] = - c  + c/2*Limiter(r[(j-1)%nx]);
    return scipy.linalg.solve(B,Phi)
    

# ------------------------------------------------------------------------------------
###                          ••••• Composed Schemes •••••                          ###
# ------------------------------------------------------------------------------------
### the schemes below, consist of composing the above schemes.
    
    
def TRBDF2(nx,c,Phi):
    gamma = 2-np.sqrt(2)
    Phi1 = np.zeros([nx]); beta = np.zeros([nx]);
    Phi1 = CN(nx,gamma*c,Phi)
    beta[:] = (1/(gamma))*(1/(2-gamma))*Phi1[:] - (1/(gamma))*(1/(2-gamma))*(1-gamma)**2*Phi[:]
    return BE(nx,(1-gamma)/(2-gamma)*c, beta )

def SDIRK2(nx,c,Phi):
    return CN(nx,0.5*c,CN(nx,0.5*c,Phi))  
    
def TRBDF2K(nx,c,Phi):
    gamma = 2-np.sqrt(2); Phi1 = np.zeros([nx]); beta = np.zeros([nx]);
    Phi1 = CNK(nx,gamma*c,Phi)
    beta[:] = (1/(gamma))*(1/(2-gamma))*Phi1[:] - (1/(gamma))*(1/(2-gamma))*(1-gamma)**2*Phi[:]
    return BEK(nx,(1-gamma)/(2-gamma)*c, beta )

def SDIRK2K(nx,c,Phi):
    return CNK(nx,0.5*c,CNK(nx,0.5*c,Phi))


    
def CN(nx,c,Phi):
    return FE(nx,0.5*c,BE(nx,0.5*c,Phi))
def CNK(nx,c,Phi):
    return FEK(nx,0.5*c,BEK(nx,0.5*c,Phi))   
    
def SDIRK4K(nx,c,Phi):
    return SDIRK2K(nx,0.5*c,SDIRK2K(nx,0.5*c,Phi))
def SDIRK8K(nx,c,Phi):
    return SDIRK4K(nx,0.5*c,SDIRK4K(nx,0.5*c,Phi))
def SDIRK16K(nx,c,Phi):
    return SDIRK8K(nx,0.5*c,SDIRK8K(nx,0.5*c,Phi))


# ------------------------------------------------------------------------------------
###                          ••••• Indicator functions •••••                          ###
# ------------------------------------------------------------------------------------

def indEXP(x):
    EXP = np.zeros_like(x)
    for i in range(0,len(x)):
        if (x[i]>0.55):
            EXP[i] = 1
        if (x[i]<0.45):
            EXP[i] = 1
    return EXP
    
def indIMP(x):
    return 1 - indEXP(x)

    # ------------------------------------------------------------------------------------
    ###                          ••••• Blended schemes •••••                          ###
    # ------------------------------------------------------------------------------------
    

# ------------------------------------------------------------------------------------
###                          ••••• FEBE methods •••••                          ###
# ------------------------------------------------------------------------------------
def viBEFE(nx,c,Phi,IMP,EXP):
    ## volume indicator ##
    Phiprime = np.zeros(nx)
    
    Phiprime[:] = Phi[:]  - c*EXP*(Phi[:] - np.roll(Phi[:],1)) 
    
    B = np.identity(nx)
    for j in range(0,nx):
        B[j,(j-1)%nx] = -IMP[j]*c;
        B[j,j]        = 1 + IMP[j]*c;
        
    return np.linalg.solve(B,Phiprime)
    
def viBEFEK(nx,c,Phi,IMP,EXP):
    ## Volume indicator ##
    ep = 10**(-8)
    r = np.zeros(nx); rr = np.zeros(nx);
    Phihalf = np.zeros(nx)
    r =  ( np.roll(Phi[:],0) - np.roll(Phi[:],1) ) / ( np.roll(Phi[:],-1) - np.roll(Phi[:],0) +ep)
    for j in range(0,nx):
        rr[j] = Limiter(r[j%nx])
    Phihalf[:]  = (Phi[:] + 1/2*rr*(  np.roll(Phi[:],-1) - np.roll(Phi[:],0) )) ### second order unstable method
    Phi = Phi[:] - c*( np.roll(Phihalf[:],0) - np.roll(Phihalf[:],1) )*EXP
    B = np.identity(nx)
    for j in range(0,nx):
        B[j,j] = 1  + c*IMP[j] - c/2*Limiter(r[j%nx])*IMP[j] - c/2*Limiter(r[(j-1)%nx])*IMP[(j)%nx] ;
        B[j,(j+1)%nx] = c/2*Limiter(r[j%nx])*IMP[j];
        B[j,(j-1)%nx] = - c*IMP[(j)%nx]  + c/2*Limiter(r[(j-1)%nx])*IMP[(j)%nx];
    return np.linalg.solve(B,Phi)
    

#def fiBEFE(nx,c,Phi,IMP,EXP):
#    ## face indicator ##
#    Phihalf = np.zeros(nx)
#    Phihalf = Phi*EXP ## decide whether the face contributes implicitly or explicitly
#    Phi[:] = Phi[:] - c*(Phihalf[:] - np.roll(Phihalf[:],1)) 
#    
#    B = np.zeros([nx,nx])
#    for j in range(0,nx):
#        B[j,(j-1)%nx] =   - IMP[(j-1)%nx]*c;
#        B[j,j]        = 1  + IMP[j]*c;
#    return np.linalg.solve(B,Phi)
    
def fiBEFE(nx,c,Phi,IMP,EXP):
    """Hilary's face indicator method for implicit-explicit blended advection with multiple
    iterations. First-order upwind in space, first-order forward or backward in time. 
    Input argument nx is not needed and should be removed.
    c is the Courant number
    Phi is the dependent variable that is advected.
    IMP is the locations where it is implicit.
    EXP is not used and should be removed
    Phi  at the next time step is returned."""
    
    nx = len(Phi)
    EXP = 1 - IMP

    phiOld = Phi.copy()
    
    B = np.zeros([nx,nx])
    for j in range(0,nx):
        B[j,(j-1)%nx] =   - IMP[(j-1)%nx]*c;
        B[j,j]        = 1  + IMP[j]*c;

    for iter in range(1):
        Phi = phiOld - c*(EXP*Phi - np.roll(EXP*Phi,1))
        Phi = np.linalg.solve(B,Phi)

    return Phi

def fiBEFEK(nx,c,Phi,IMP,EXP):
    ## face indicator ##
    ep = 10**(-8)
    r = np.zeros(nx); rr = np.zeros(nx);
    Phihalf = np.zeros(nx)
    r =  ( np.roll(Phi[:],0) - np.roll(Phi[:],1) ) / ( np.roll(Phi[:],-1) - np.roll(Phi[:],0) +ep)
    for j in range(0,nx):
        rr[j] = Limiter(r[j%nx])
    Phihalf[:]  = (Phi[:] + 1/2*rr*(  np.roll(Phi[:],-1) - np.roll(Phi[:],0) ))*EXP ### second order unstable method
    Phi = Phi[:] - c*( np.roll(Phihalf[:],0) - np.roll(Phihalf[:],1) )
    B = np.identity(nx)
    for j in range(0,nx):
        B[j,j] = 1  + c*IMP[j] - c/2*Limiter(r[j%nx])*IMP[j] - c/2*Limiter(r[(j-1)%nx])*IMP[(j-1)%nx] ;
        B[j,(j+1)%nx] = c/2*Limiter(r[j%nx])*IMP[j];
        B[j,(j-1)%nx] = - c*IMP[(j-1)%nx]  + c/2*Limiter(r[(j-1)%nx])*IMP[(j-1)%nx];
    return np.linalg.solve(B,Phi)
    


    # ------------------------------------------------------------------------------------
    ###                          ••••• SSP104BE methods •••••                          ###
    # ------------------------------------------------------------------------------------

def viBESSP104(nx,c,Phi,IMP,EXP):
    ## Volume indicator ##
    Phi1 = np.zeros([nx]);Phi2 = np.zeros([nx]);
    Phi1[:] = Phi[:]
    for k in range(0,4):
        Phi1[:] = Phi1[:] - 1/6*c*EXP*( Phi1[:] - np.roll(Phi1[:],1) )
    Phi2 = 3/5*Phi[:] +2/5*Phi1[:] - 1/15*c*EXP*( Phi1[:] - np.roll(Phi1[:],1) )
    for k in range(0,4):
        Phi2[:] = Phi2[:]  - 1/6*c*EXP*( Phi2[:] - np.roll(Phi2[:],1) )
    Phi2[:] = 1/25*Phi[:] + 9/25*Phi1[:] +3/5*Phi2[:]  \
     -3/50*c*EXP*( Phi1[:] - np.roll(Phi1[:],1) )          \
     -1/10*c*EXP*( Phi2[:] - np.roll(Phi2[:],1) )    
    
    ## implicit part
    
    B = np.zeros([nx,nx])
 
    for j in range(0,nx):
        B[j,j] = 1  + c*IMP[j];
        B[j,(j-1)%nx] = -c*IMP[j];
    return np.linalg.solve(B,Phi2)

def viBESSP104Kk(nx,c,Phi,IMP,EXP):
    ## Volume indicator ##
    ##Implements Volume indicator method high order for ssprk, and low order for BE
    ## stable for cfl <0.5*6
    ep = 10**(-8)
    r = np.zeros(nx); rr = np.zeros(nx);
    Phi1 = np.zeros([nx]);Phi2 = np.zeros([nx]);
    Phihalf = np.zeros(nx);Phihalf1 = np.zeros(nx);
    
    Phi1[:] = Phi[:]
    for k in range(0,4):
        r =  ( np.roll(Phi[:],0) - np.roll(Phi[:],1) ) / ( np.roll(Phi[:],-1) - np.roll(Phi[:],0) +ep)
        for j in range(0,nx):
            rr[j] = Limiter(r[j%nx])
        Phihalf[:]  = (Phi1[:] + 1/2*rr*(  np.roll(Phi1[:],-1) - np.roll(Phi1[:],0) )) ### second order unstable method
        Phi1[:] = Phi1[:] - 1/6*c*( Phihalf[:] - np.roll(Phihalf[:],1) )*EXP
        
        
    Phihalf[:]  = (Phi1[:] + 1/2*rr*(  np.roll(Phi1[:],-1) - np.roll(Phi1[:],0) ))
    Phihalf1[:]  = Phihalf[:];
    
    
    Phi2 = 3/5*Phi[:] +2/5*Phi1[:] - 1/15*c*( Phihalf[:] - np.roll(Phihalf[:],1) )
    for k in range(0,4):
        r =  ( np.roll(Phi2[:],0) - np.roll(Phi2[:],1) ) / ( np.roll(Phi2[:],-1) - np.roll(Phi2[:],0) +ep)
        for j in range(0,nx):
            rr[j] = Limiter(r[j%nx])
        Phihalf[:]  = (Phi2[:] + 1/2*rr*(  np.roll(Phi2[:],-1) - np.roll(Phi2[:],0) ))
        
        Phi2[:] = Phi2[:]  - 1/6*c*( Phihalf[:] - np.roll(Phihalf[:],1) )*EXP
        
    Phihalf[:]  = (Phi2[:] + 1/2*rr*(  np.roll(Phi2[:],-1) - np.roll(Phi2[:],0) ))  
    Phi2[:] = 1/25*Phi[:] + 9/25*Phi1[:] +3/5*Phi2[:]  \
     -3/50*c*( Phihalf1[:] - np.roll(Phihalf1[:],1) )*EXP           \
     -1/10*c*( Phihalf[:] - np.roll(Phihalf[:],1) )*EXP    
    
    ## question, what r do we use here? 
    B = np.identity(nx)
    for j in range(0,nx):
        B[j,j] = 1  + c*IMP[j];
        B[j,(j-1)%nx] = - c*IMP[j];
    return np.linalg.solve(B,Phi2)
    
def viBESSP104K(nx,c,Phi,IMP,EXP):
    ## Volume indicator ##
    ##Implements Volume indicator method high order for ssprk, and low order for BE
    ## stable for cfl <0.5*6
    ep = 10**(-8)
    r = np.zeros(nx); rr = np.zeros(nx);
    Phi1 = np.zeros([nx]);Phi2 = np.zeros([nx]);
    Phihalf = np.zeros(nx);Phihalf1 = np.zeros(nx);
    
    Phi1[:] = Phi[:]
    for k in range(0,4):
        r =  ( np.roll(Phi[:],0) - np.roll(Phi[:],1) ) / ( np.roll(Phi[:],-1) - np.roll(Phi[:],0) +ep)
        for j in range(0,nx):
            rr[j] = Limiter(r[j%nx])
        Phihalf[:]  = (Phi1[:] + 1/2*rr*(  np.roll(Phi1[:],-1) - np.roll(Phi1[:],0) )) ### second order unstable method
        Phi1[:] = Phi1[:] - 1/6*c*( Phihalf[:] - np.roll(Phihalf[:],1) )*EXP
        
        
    Phihalf[:]  = (Phi1[:] + 1/2*rr*(  np.roll(Phi1[:],-1) - np.roll(Phi1[:],0) ))
    Phihalf1[:]  = Phihalf[:];
    
    
    Phi2 = 3/5*Phi[:] +2/5*Phi1[:] - 1/15*c*( Phihalf[:] - np.roll(Phihalf[:],1) )
    for k in range(0,4):
        r =  ( np.roll(Phi2[:],0) - np.roll(Phi2[:],1) ) / ( np.roll(Phi2[:],-1) - np.roll(Phi2[:],0) +ep)
        for j in range(0,nx):
            rr[j] = Limiter(r[j%nx])
        Phihalf[:]  = (Phi2[:] + 1/2*rr*(  np.roll(Phi2[:],-1) - np.roll(Phi2[:],0) ))
        
        Phi2[:] = Phi2[:]  - 1/6*c*( Phihalf[:] - np.roll(Phihalf[:],1) )*EXP
        
    Phihalf[:]  = (Phi2[:] + 1/2*rr*(  np.roll(Phi2[:],-1) - np.roll(Phi2[:],0) ))  
    Phi2[:] = 1/25*Phi[:] + 9/25*Phi1[:] +3/5*Phi2[:]  \
     -3/50*c*( Phihalf1[:] - np.roll(Phihalf1[:],1) )*EXP           \
     -1/10*c*( Phihalf[:] - np.roll(Phihalf[:],1) )*EXP    
    
    ## question, what r do we use here? 
    r= ( np.roll(Phi2[:],0) - np.roll(Phi2[:],1) ) / ( np.roll(Phi2[:],-1) - np.roll(Phi2[:],0) +ep)
    B = np.identity(nx)
    for j in range(0,nx):
        B[j,j] = 1  + c*IMP[j] - c/2*Limiter(r[j%nx])*IMP[j] - c/2*Limiter(r[(j-1)%nx])*IMP[(j)] ;
        B[j,(j+1)%nx] = c/2*Limiter(r[j%nx])*IMP[j];
        B[j,(j-1)%nx] = - c*IMP[(j)%nx]  + c/2*Limiter(r[(j-1)%nx])*IMP[(j)%nx];
    return np.linalg.solve(B,Phi2)
    

def fiBESSP104Kk(nx,c,Phi,IMP,EXP):
    ## Face indicator ##
    ep = 10**(-8)
    r = np.zeros(nx); rr = np.zeros(nx);
    Phi1 = np.zeros([nx]);Phi2 = np.zeros([nx]);
    Phihalf = np.zeros(nx);Phihalf1 = np.zeros(nx);
    
    Phi1[:] = Phi[:]
    for k in range(0,4):
        r =  ( np.roll(Phi[:],0) - np.roll(Phi[:],1) ) / ( np.roll(Phi[:],-1) - np.roll(Phi[:],0) +ep)
        for j in range(0,nx):
            rr[j] = Limiter(r[j%nx])
        Phihalf[:]  = (Phi1[:] + 1/2*rr*(  np.roll(Phi1[:],-1) - np.roll(Phi1[:],0) ))*EXP ### second order unstable method
        Phi1[:] = Phi1[:] - 1/6*c*( Phihalf[:] - np.roll(Phihalf[:],1) )
        
        
    Phihalf[:]  = (Phi1[:] + 1/2*rr*(  np.roll(Phi1[:],-1) - np.roll(Phi1[:],0) ))*EXP
    Phihalf1[:]  = Phihalf[:];
    
    
    Phi2 = 3/5*Phi[:] +2/5*Phi1[:] - 1/15*c*( Phihalf[:] - np.roll(Phihalf[:],1) )
    for k in range(0,4):
        r =  ( np.roll(Phi2[:],0) - np.roll(Phi2[:],1) ) / ( np.roll(Phi2[:],-1) - np.roll(Phi2[:],0) +ep)
        for j in range(0,nx):
            rr[j] = Limiter(r[j%nx])
        Phihalf[:]  = (Phi2[:] + 1/2*rr*(  np.roll(Phi2[:],-1) - np.roll(Phi2[:],0) ))*EXP
        
        Phi2[:] = Phi2[:]  - 1/6*c*( Phihalf[:] - np.roll(Phihalf[:],1) )
        
    Phihalf[:]  = (Phi2[:] + 1/2*rr*(  np.roll(Phi2[:],-1) - np.roll(Phi2[:],0) ))*EXP   
    Phi2[:] = 1/25*Phi[:] + 9/25*Phi1[:] +3/5*Phi2[:]  \
     -3/50*c*( Phihalf1[:] - np.roll(Phihalf1[:],1) )          \
     -1/10*c*( Phihalf[:] - np.roll(Phihalf[:],1) )    
    
    ## question, what r do we use here? 
    B = np.identity(nx)
    for j in range(0,nx):
        B[j,j] = 1  + c*IMP[j];
        B[j,(j-1)%nx] = - c*IMP[(j-1)%nx];
    return np.linalg.solve(B,Phi2)


    
    
    
# ------------------------------------------------------------------------------------
###                          ••••• Plotting tools •••••                            ###
# ------------------------------------------------------------------------------------
def Animate(Phi,x,i,a,dt,tv):
    plt.clf()
    plt.title(tv[i])
    plt.axis([0, 1, -0.1, 1.3])
    plt.plot(x,Phi,'b')
    plt.plot(x,IC((x-i*a*dt)%1 ),'r')
    plt.draw()
    plt.pause(0.003)
    return
    
def Total_variation(Phi):
    nx = len(Phi);
    tv = 0
    for j in range(0,nx):
        tv += abs( Phi[j] - Phi[(j-1)%nx] )
    return tv

def Measures(Phi,x,i,a,dt):
    nx = len(Phi);
    mass = np.linalg.norm(Phi,1)
    energy = np.linalg.norm(Phi,2)
    variance = 1/nx*np.linalg.norm(Phi - mass/nx,2)
    return [mass,energy,variance]
    
def Printer(Y,x,time,scheme,c,res):
    plt.clf()
    X, T = np.meshgrid(x, time)
    fig, ax = plt.subplots(1,sharex=True, sharey=True)
    cmap = plt.cm.hot

    
    #cmap = plt.cm.nipy_spectral
    levels = np.linspace(0-10e-12,1.0+10e-12,41)
    pcm = plt.contourf(X, T, Y,levels = levels,cmap = cmap,extend = "both")
    

    pcm.cmap.set_over("magenta")
    pcm.cmap.set_under('cyan')
    fig.colorbar(pcm)

    plt.ylabel('Time')
    plt.xlabel('Space')
    #levels=[-0.00000, 1.00000]
    #CS = plt.contour(X, T, Y, levels=levels,colors='k')
    
    #plt.clabel(CS,fontsize=7)
    plt.title(''+str(scheme)+' SpaceTimePlot at cfl '+str(c)+'')
    plt.savefig(''+str(scheme)+'-SpaceTimePlot.png',dpi = res)
    
    return 
    
def ICSaver(Phi,x,c,scheme,IMP,EXP,res):
    plt.clf()
    plt.title('Initial Conditions')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.axis([0, 1, -0.1, 1.3])
    plt.plot(x,-10*EXP,'.',color="skyblue",label = 'explicit')
    plt.plot(x,-10*IMP,'.',color='g',label = 'implicit')
    p1 = plt.fill_between( x, 10*EXP, color="skyblue", alpha=0.4)
    p2 = plt.fill_between( x, 10*IMP, color='g', alpha=0.4)
    plt.plot(x,IC(x),'k',label = 'initial conditions')
    plt.legend((p1, p2), ('Explicit', 'Implicit'))
    plt.savefig('Scheme'+str(scheme)+'InitialConditions.png',dpi = res)
    plt.clf()
    return 
    
def AnimationSaver(Phi,x,i,a,dt,tv,c,scheme):
    plt.clf()
    plt.title(tv[i])
    plt.axis([0, 1, -0.1, 1.3])
    plt.plot(x,Phi,'b')
    plt.plot(x,IC((x-i*a*dt)%1 ),'r')
    if (i%100 == 0):
        plt.savefig('Scheme'+str(scheme)+'frame-'+str(int(i/100))+'.png')
    return     
    
def giffMaker2(scheme):
    path = '/Users/jmw418/Desktop/Python_Blended_Numerical_Schemes_1d/One_dimensional_split_schemes/Harten_volume_indicater'
    image_folder = os.fsencode(path)

    filenames = []

    for file in os.listdir(image_folder):
        filename = os.fsdecode(file)
        if filename.startswith( ('Scheme'+str(scheme)+'') ):
            filenames.append(filename)

    filenames.sort() # this iteration technique has no built in order, so sort the frames
    natsort.natsorted(filename)
    images = [imageio.imread(f) for f in filenames] 
    imageio.mimsave(os.path.join('Scheme'+str(scheme)+'movie.gif'), images, duration = 0.04) # modify duration as needed
    
    #plt.pause(0.003)



if __name__ == "__main__":
    # execute only if run as a script
    main()
    




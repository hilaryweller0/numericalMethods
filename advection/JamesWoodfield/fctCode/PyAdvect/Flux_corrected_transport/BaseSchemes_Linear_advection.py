#!/usr/local/bin/python3 
"""
@author: jmw418
# -*- coding: utf-8 -*-
## """
###Packages###

import numpy as np
import numpy as np
import scipy.linalg
from numpy import linalg 
import scipy.sparse.linalg as sp
from scipy import fftpack as ft
import matplotlib.pyplot as plt
import time
from time import process_time 
import copy

def Total_variation(Phi):
    nx = len(Phi);
    tv = 0
    for j in range(0,nx):
        tv += abs( Phi[j] - Phi[(j-1)%nx] )
    return tv
# ------------------------------------------------------------------------------------
###                          ••••• Limiters •••••                ###
# ------------------------------------------------------------------------------------

def HO(r):
    return 1
def LO(r):
    return 0
    
def min_mod(r):
    return max(0,min(1,r)) 
def van_Albada(r):
    return (r**2+r)/(r**2+1)
def Korem(r):
    return max(0,min(2,2/3+r/3,2*r))
def VanL(r):
    return (r+abs(r))/(1+abs(r))
def superbee(r):
    return max(0,min(2*r,1),min(r,2))
    
    
    
def Limiter(r):
    ### general method of lines limiter, suitable for nonlinear and implicit
    #cfl restriction typically [0,0.5]
    return min_mod(r)
    
def Limiter_LW(r,c):
    ### standard modification to lax wendroff flux for linear advection,
    # only suitable for forwards euler steps.
    # allows cfl in [0,1] doubles timestep. 
    return Limiter(r)*(1-c)


# ------------------------------------------------------------------------------------
###                •••••  Schemes •••••                ###
# ------------------------------------------------------------------------------------
        
        


# ------------------------------------------------------------------------------------
###                •••••  Low order Schemes •••••                ###
# ------------------------------------------------------------------------------------
        
def BELL(c,Phi):
    '''Backward Euler for Linear advection, at Low spatial order
    inputs:
    c is the Courant number
    Phi is the dependent variable that is advected. '''
    nx = len(Phi); A = np.zeros([nx,nx]); 
    for j in range(0,nx):
        A[j,j] = 1  + c;
        A[j,(j-1)%nx] = -c;
    return np.linalg.solve(A,Phi)
    
def BELL_fast(c,Phi): 
    '''alternative implementation of the above, 10 times faster for large matrices '''
    nx = len(Phi)
    Circ = np.zeros(nx); Circ[0] = 1+c;Circ[1] = -c
    ## we can invert circulant matrices by division in fourier space
    return ft.ifft(ft.fft(Phi)/ft.fft(Circ))

def FELL(c,Phi):
    '''Forward Euler Linear Low
    First-order upwind in space, first-order forward in time.
    inputs:
    c is the Courant number
    Phi is the dependent variable that is advected. '''
    return Phi - c* (Phi - np.roll(Phi,1))
    

### ••••• Higher order explicit Corrections•••  ###
##  below are two options that create the finite difference array of certain order.
def Centered(Phi,pts):
    """Fornberg, Bengt., 1988. Generation of finite difference formulas on arbitrarily spaced grids. """
    '''input your, number of interpolating points, 1 gives the standard centered difference schemes'''
    aax = np.asarray(np.arange(-pts, pts+1))
    print('aax=',aax)
    ax = np.asarray(finite_diff_weights(1, aax, 0)[-1][-1])
    print('ax=',ax)
    nn = len(ax); print(nn);
    f = np.zeros(len(Phi))
    for i in range(0,nn):
        print("i,aax[i], ax[i]",i,aax[i],ax[i])
        f = f + np.roll(Phi,-aax[i])*ax[i]
    return f
def Upwind(Phi,pts):
    """Fornberg, Bengt., 1988. Generation of finite difference formulas on arbitrarily spaced grids. """
    aax = np.asarray(np.arange(-pts, pts)) ## can modify the stencil to put in upwind bias, eg remove the last point in the stencil
    print('aax=',aax)
    ax = np.asarray(finite_diff_weights(1, aax, 0)[-1][-1])
    print('ax=',ax)
    nn = len(ax); print(nn);
    f = np.zeros(len(Phi))
    for i in range(0,nn):
        print("i,aax[i], ax[i]",i,aax[i],ax[i])
        f = f + np.roll(Phi,-aax[i])*ax[i]
    return f
#print(Centered(np.asarray([0.3,0.5,0.7,1]),1))

# ------------------------------------------------------------------------------------
###            ••••• high order Schemes •••••                ###
# the idea is these are put into the value for phi-half-minus
# ------------------------------------------------------------------------------------
### give the phi-Half, ###

def MUSCL_linear(Phi):
    '''MUSCL-kappa scheme - correction on upwind, not limited'''
    return Phi + 1/4*( np.roll(Phi,-1) - np.roll(Phi,1) )
    
def MUSCL_limited(Phi):
    '''MUSCL-kappa scheme - limited theoretical stability of cfl 0.5,
     linear equation version. One could compactify the stencil.'''
    
    kappa = 1/3 # -1(second order upwind), 0(From),1/2(QUick), 1/3(third order upwind), 1 Centered.
    ep = 10**(-8)
    nx = len(Phi)
    r = np.zeros(nx); rr = np.zeros(nx); rrr = np.zeros(nx);
    PhihalfMinus = np.zeros(nx); # this creation could be suppressed
    r = ((( Phi - np.roll(Phi,1) +ep) / ( np.roll(Phi,-1) - Phi +ep))) 
    for j in range(0,nx):
        rr[j]  = Limiter(r[j%nx])
        rrr[j] = Limiter(1/r[j%nx])
    PhihalfMinus = Phi + (1-kappa)/4*rrr*(  Phi - np.roll(Phi,1) ) \
    +  (1+kappa)/4*rr*( np.roll(Phi,-1) - Phi  )  
    return PhihalfMinus
    

def WENO5_L(Phi):
    '''Weno5 correction to upwind, it is made up of a convex combination of the 
    Eno3 stencils.
    Important detail: RONG WANG AND RAYMOND J. SPITERI 
    Show that it is linearly unstable for forward euler, 
    but recoverable with other timestepping techniques 3 step'''
    ep = 10**(-6)
    lam1 = 1/10; lam2 = 3/5; lam3 = 3/10
    nx = len(Phi)
    f1 = np.zeros(nx);f2 = np.zeros(nx);f3 = np.zeros(nx);
    b1 = np.zeros(nx);b2 = np.zeros(nx);b3 = np.zeros(nx);
    ww1 = np.zeros(nx);ww2 = np.zeros(nx);ww3 = np.zeros(nx);
    w1 = np.zeros(nx);w2 = np.zeros(nx);w3 = np.zeros(nx);  
    
    f1 = 1/3*np.roll(Phi,2) - 7/6*np.roll(Phi,1) +11/6*Phi
    f2 = -1/6*np.roll(Phi,1) +5/6*Phi + 1/3*np.roll(Phi,-1) 
    f3 = 1/3*Phi +5/6*np.roll(Phi,-1)  - 1/6*np.roll(Phi,-2)
     
    b1 = 13/12*(np.roll(Phi,2) - 2*np.roll(Phi,1) + Phi)**2 \
    +1/4*(np.roll(Phi,2) - 4*np.roll(Phi,1) + 3*Phi)**2
    b2 = 13/12*(np.roll(Phi,1) - 2*np.roll(Phi,0) + np.roll(Phi,-1))**2 \
    +1/4*(np.roll(Phi,1) - np.roll(Phi,-1))**2
    b3 = 13/12*( Phi - 2*np.roll(Phi,-1) + np.roll(Phi,-2) )**2 \
    +1/4*( 3*Phi  - 4*np.roll(Phi,-1) + np.roll(Phi,-2) )**2
    
    ww1 = lam1/((ep+b1)**2)
    ww2 = lam1/((ep+b2)**2)
    ww2 = lam1/((ep+b3)**2)
    
    w1 = ww1/(ww1+ww2+ww3)
    w2 = ww2/(ww1+ww2+ww3)
    w3 = ww3/(ww1+ww2+ww3)
    
    fhalf = w1*f1+w2*f2+w3*f3
    #- np.roll((fhalf - Phi),1)
    return fhalf


# ------------------------------------------------------------------------------------
###                ••••• Implicit high order method •••••               ###
## linearised flux limiters.
# ------------------------------------------------------------------------------------

def BEMUSCL_L(c,Phi):
    '''Backward euler, MUSCL, we linearise the limiter. slightly violates monotonicity.
    2nd order in space '''
    "Helen C. Yee - conservative linearisation."
    kappa = 1/3 # -1, 0, 1/3, 1
    ep = 10**(-8)
    nx = len(Phi)
    r = np.zeros(nx); rr = np.zeros(nx); rrr = np.zeros(nx);
    PhihalfMinus = np.zeros(nx); PhihalfPlus = np.zeros(nx);  #not needed 
    r =  ( Phi - np.roll(Phi,1) +ep) / ( np.roll(Phi,-1) - Phi +ep)
    #r = 1/r
    for j in range(0,nx):
        rr[j]  = Limiter(r[j%nx])
        rrr[j] = Limiter(1/r[j%nx])
    B = np.identity(nx)
    for j in range(0,nx):
        B[j,j] = 1 + c + c*(1-kappa)/4*rrr[j] - c*(1+kappa)/4*rr[j] \
         - c*(1+kappa)/4*rr[(j-1)%nx];
        B[j,(j+1)%nx] =  c*(1+kappa)/4*rr[j];
        B[j,(j-1)%nx] =  - c*(1-kappa)/4*rrr[j] \
        -c - c*(1-kappa)/4*rrr[(j-1)%nx] + c*(1+kappa)/4*rr[(j-1)%nx];
        B[j,(j-2)%nx] = c*(1-kappa)/4*rrr[(j-1)%nx]
        
    return scipy.linalg.solve(B,Phi)

    
# -------------------------------------------------------------------------
###             ••••• Advanced FE options. •••••             ###
#--------------------------------------------------------------------------

def FE_centered_ho(c,Phi,pts):
    '''Forwards euler Muscl kappa scheme, for linear advection.
    inputs:
    c is the Courant number
    Phi is the dependent variable that is advected.'''
    Phihalf = Phi + Centered(Phi,pts)
    return  Phi - c*(Phihalf - np.roll(Phihalf,1))
    
def FE_upwind_ho(c,Phi,pts):
    '''Forwards euler Muscl kappa scheme, for linear advection.
    inputs:
    c is the Courant number
    Phi is the dependent variable that is advected.'''
    Phihalf = Phi + Upwind(Phi,pts)
    return Phi - c*(Phihalf - np.roll(Phihalf,1))
    
def FEMUSCL_L(c,Phi):
    '''Forwards euler Muscl kappa scheme, for linear advection.
    inputs:
    c is the Courant number
    Phi is the dependent variable that is advected.'''
    Phihalf  = MUSCL_limited(Phi)
    return Phi - c*( Phihalf - np.roll(Phihalf,1) )
    
    
def FEWENO5_L(c,Phi):
    '''Forwards Euler , Weno5, Linear advection (unstable)'''
    Phihalf = WENO5_L(Phi)
    return Phi - c*( Phihalf - np.roll(Phihalf,1) )
    
def FEMUSCL_LW(c,Phi):
    '''Forwards euler Muscl kappa scheme, for linear advection.
    inputs:
    c is the Courant number
    Phi is the dependent variable that is advected.'''
    Phihalf  = MUSCL_limited(Phi)*(1-c)  - Phi*(1-c) + Phi
    return Phi - c*( Phihalf - np.roll(Phihalf,1) )
    
    
    
# ------------------------------------------------------------------------------------
###                   ••••• Advanced explicit RK options •••••                     ###
# ------------------------------------------------------------------------------------
    
def SSP33WENO5_L(c,Phi):
    '''linear advection'''
    return 1/3*Phi+2/3*FEWENO5_L(c,3/4*Phi+1/4*FEWENO5_L(c,FEWENO5_L(c,Phi)))
    
def SSP33MUSCL_L(c,Phi):
    return 1/3*Phi+2/3*FEMUSCL_L(c,3/4*Phi+1/4*FEMUSCL_L(c,FEMUSCL_L(c,Phi)))   

def FEMUSCL_LW_2(c,Phi):
    ## multiple stages, required, increases the cfl number, and stops the step phenomenom.
    return FEMUSCL_LW(0.5*c,FEMUSCL_LW(0.5*c,Phi))
    
def SSPRK3_4stage(c,Phi):
    ## one more stage than, ssp33, but has c = 2. so that C_eff = 2/4 = 0.5 rather than 1/3.
    # 1.5 times more computationally efficient.
    
    
    Phi1 = FELL(0.5*c,Phi)
    Phi2 = FELL(0.5*c,Phi1)
    Phi3 = 2/3*Phi + 1/3*FELL(0.5*c,Phi2)
    Phi3 = FELL(0.5*c,Phi3)
    return Phi3
    
def SSPRK3_4stage_C(c,Phi):
    return FELL(0.5*c,2/3*Phi + 1/3*FELL(0.5*c,FELL(0.5*c,FELL(0.5*c,Phi))))
    
def SSP33_L(c,Phi):
    '''linear advection'''
    return 1/3*Phi+2/3*FELL(c,3/4*Phi+1/4*FELL(c,FELL(c,Phi)))
    
def SSP104MUSCL_L(c,Phi):
    '''Extension of FEMUSCL_L to ssp104 
    inputs:
    c is the Courant number
    Phi is the dependent variable that is advected.'''
    ## stable for cfl <0.5*6*max(ffp)
    ep = 10**(-8)
    nx = len(Phi)
    r = np.zeros(nx); rr = np.zeros(nx);
    Phi1 = np.zeros([nx]);Phi2 = np.zeros([nx]);
    Phihalf = np.zeros(nx);Phihalf1 = np.zeros(nx);
    
    Phi1[:] = Phi[:]
    for k in range(0,4):
        Phi1 = FEMUSCL_L(c/6,Phi1)
        
    Phihalf[:] =  MUSCL_limited(Phi1)
    Phihalf1[:]  = Phihalf[:]; ## this stores the values for later
    
    Phi2 = 3/5*Phi[:] +2/5*Phi1[:] - 1/15*c*( Phihalf[:] - np.roll(Phihalf[:],1))
    
    for k in range(0,4):
        Phi2 = FEMUSCL_L(c/6,Phi2)
        
    Phihalf[:]  = MUSCL_limited(Phi2)  
    
    Phi2[:] = 1/25*Phi + 9/25*Phi1 +3/5*Phi2  \
     -3/50*c*( Phihalf1 - np.roll(Phihalf1,1) )  \
     -1/10*c*( Phihalf - np.roll(Phihalf,1) ) 
    
    return Phi2
    
def SSP104WENO5_L(c,Phi):
    '''Extension of FEweno5 to ssp104 
    inputs:
    c is the Courant number
    Phi is the dependent variable that is advected.'''
    ## stable for cfl <0.5*6*max(ffp)
    ep = 10**(-8)
    nx = len(Phi)
    r = np.zeros(nx); rr = np.zeros(nx);
    Phi1 = np.zeros([nx]);Phi2 = np.zeros([nx]);
    Phihalf = np.zeros(nx);Phihalf1 = np.zeros(nx);
    
    Phi1[:] = Phi[:]
    for k in range(0,4):
        Phi1 = FEWENO5_L(c/6,Phi1)
        
    Phihalf[:] =  WENO5_L(Phi1)
    Phihalf1[:]  = Phihalf[:]; ## this stores the values for later
    
    Phi2 = 3/5*Phi[:] +2/5*Phi1[:] - 1/15*c*( Phihalf[:] - np.roll(Phihalf[:],1))
    
    for k in range(0,4):
        Phi2 = FEWENO5_L(c/6,Phi2)
        
    Phihalf[:]  = WENO5_L(Phi2)  
    
    Phi2[:] = 1/25*Phi + 9/25*Phi1 +3/5*Phi2  \
     -3/50*c*( Phihalf1 - np.roll(Phihalf1,1) )  \
     -1/10*c*( Phihalf - np.roll(Phihalf,1) ) 
    
    return Phi2
    
    
def SSPRK2_2s(c,Phi):
    s = round(c)+1
    c = c/(s);
    for k in range(0,s):
        Phi1 = FELL(c,Phi) 
        Phi = Phi + 0.5*((FELL(c,Phi)-Phi) + (FELL(c,Phi1) - Phi1))
    return Phi
    
def RK4Spectral_L(Phi,dt,k,a):## this scheme should add in viscosity.
    dt = dt
    k1 =  dt*ft.ifft(a*k*1j*ft.fft( -(Phi) ))  
    #print(k1)
    k2 = np.real( dt*ft.ifft(a*k*1j*ft.fft( -(Phi+0.5*k1) )) )
    k3 = np.real( dt*ft.ifft(a*k*1j*ft.fft( -(Phi+0.5*k2) )) )   
    k4 = np.real( dt*ft.ifft(a*k*1j*ft.fft( -(Phi+k3) )) )      
       #FE in fourier space    
    Phi = Phi + 1/6*( k1 + 2*k2 + 2*k3 + k4 ) 
    return Phi    
    

    
# -------------------------------------------
###          ••••• Advanced implicit RK options •••••     ###
###                   ••••• low order in space •••••     ###
#---------------------------------------------
    
def ThetaLL(c,Phi,theta):
    return BELL( theta*c, FELL((1-theta)*c,Phi) )
    
def CNLL(c,Phi):
    return ThetaLL(c,Phi,0.5)
    
def TRBDF2LL(c,Phi):
    nx = len(Phi)
    gamma = 2-np.sqrt(2) ## chooses staggering
    Phi1 = np.zeros([nx]); beta = np.zeros([nx]);
    Phi1 = CNLL(gamma*c,Phi)
    beta = (1/(gamma))*(1/(2-gamma))*Phi1 - (1/(gamma))*(1/(2-gamma))*(1-gamma)**2*Phi
    return BELL((1-gamma)/(2-gamma)*c, beta )
    
    
def SSPIRK2_2s_L(c,Phi):
    s = round(c/2)+1 ## decide number of stages
    for k in range(0,s):
        Phi = CNLL(c/s,Phi)
    return Phi
    
# ------------------------------------------------------------------------------------
###                   ••••• Advanced implicit RK options •••••                     ###
###                   ••••• high order implementation •••••                     ###
# ------------------------------------------------------------------------------------
    
def ThetaMUSCL_L(c,Phi,theta):
    return BEMUSCL_L( theta*c, FEMUSCL_L((1-theta)*c,Phi) )
    
def CNMUSCL_L(c,Phi):
    return ThetaMUSCL_L(c,Phi,0.5)
    
def TRBDF2MUSCL_L(c,Phi):
    nx = len(Phi)
    gamma = 2-np.sqrt(2) ## chooses staggering
    Phi1 = np.zeros([nx]); beta = np.zeros([nx]);
    Phi1 = CNMUSCL_L(gamma*c,Phi)
    beta = (1/(gamma))*(1/(2-gamma))*Phi1 - (1/(gamma))*(1/(2-gamma))*(1-gamma)**2*Phi
    return BEMUSCL_L((1-gamma)/(2-gamma)*c, beta )
    

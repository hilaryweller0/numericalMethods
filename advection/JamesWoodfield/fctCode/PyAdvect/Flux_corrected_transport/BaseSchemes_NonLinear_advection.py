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
    return superbee(r)
# ------------------------------------------------------------------------------------
###              ••••• Nonlinear extension •••••               ###
# ------------------------------------------------------------------------------------
def FF(a):
    #L. Ferracina a, M.N. Spijker: Buckley–Leverett
    #3*a*a/(3*a*a + (1-a)*2 )
    ## advection. #a
    ## burgers ## a*a/2
    return a*a/2
    
def FFP(a):
    #L. Ferracina a, M.N. Spijker: Buckley–Leverett
    #6*a*(1-a)/(4*a*a-2*a+1)**2
    ## advection. #1
    ## burgers ## a
    return a
   
def LinearAdvection(L,R):
    return L
    
def Godunov_burgers(ul,ur):## Godunov numerical flux for burgers
    g = np.zeros(len(ur))
    for i in range(0,len(ur)):
        if (ul[i]<0,ur[i]<0):
            g = FF(ur)
        if (ul[i]>0,ur[i]>0):
            g = FF(ul)
        if (ul[i]<0,ur[i]>0):
            g = 0
        if (ul[i]>0,ur[i]<0):
            if (ul[i]+ur[i]>0):
                g = FF(ul)
            if (ul[i]+ur[i]<0):
                g = FF(ur)
    return g
    
def Riem(ul,ur):
    ## here we choose the solution to the reimann problem
    return Godunov_burgers(ul,ur)
    
    
    
    
def FE_NL_L(c,Phi):
    '''Forwards euler nonlinear, low order
    inputs:
    c is the Courant number
    Phi is the dependent variable that is advected.'''
    
    return Phi - c* (Riem(Phi,np.roll(Phi,-1)) - Riem(np.roll(Phi,+1),np.roll(Phi,0)) )
    
def FE_NL_Lold(c,Phi):
    '''Forwards euler nonlinear, low order
    inputs:
    c is the Courant number
    Phi is the dependent variable that is advected.'''
    return Phi - c* (FF(np.roll(Phi[:],0)) - FF(np.roll(Phi[:],1))) 


def BE_NL_L(c,Phi):
    '''Backward Euler for Nonlinear advection, at Low spatial order
    implemented with newton method
    inputs:
    c is the Courant number
    Phi is the dependent variable that is advected. 
    options: 
    tol is the tolerance of the newton method
    maxiterations is the maximum number of iterations, the newton method employs'''
    itera = 0
    maxiterations = 10
    tol = 10**(-10)
    ## initial guess old timestep:
    w = Phi.copy();
    err = 2*tol
    nx = len(w)
    beta  = np.zeros(nx);
    A = np.zeros([nx,nx])
    while (err>tol and itera<maxiterations):
        beta = -(w + c*( FF(w) - FF(np.roll(w,1)) ) - Phi);
        for j in range(0,nx):
            A[j,j] =  1 + c*FFP(w[j])
            A[j,(j-1)%nx] = - c*FFP(w[(j-1)%nx])
        
        dw = scipy.linalg.solve(A,beta)
        err  = np.linalg.norm(dw,2)
        w = dw+w 
        itera +=1
    return w 


 
# ------------------------------------------------------------------------------------
###                ••••• Higher order explicit Corrections •••••               ###
# ------------------------------------------------------------------------------------
    
def FCT_MUSCL_limited_NL(Phi):
    '''MUSCL-kappa scheme correction to upwind - limited
    theoretical stability of cfl 0.5*nl, Non-linear equation version.
    see: Hiroaki Nishikawa'''
    kappa = 1/3
    ep = 10**(-8)
    nx = len(Phi)
    r = np.zeros(nx); rr = np.zeros(nx); rrr = np.zeros(nx);
    PhihalfMinus = np.zeros(nx); PhihalfPlus = np.zeros(nx);#excessive storage atm
    r = ((( Phi - np.roll(Phi,1) +ep) / ( np.roll(Phi,-1) - Phi +ep))) 
    for j in range(0,nx):
        rr[j]  = Limiter(r[j%nx])
        rrr[j] = Limiter(1/r[j%nx])
    PhihalfMinus = Phi + (1-kappa)/4*rrr*(  Phi - np.roll(Phi,1) ) \
    +  (1+kappa)/4*rr*( np.roll(Phi,-1) - Phi  )  
    PhihalfPlus = np.roll(Phi,-1) \
    - (1-kappa)/4*np.roll(rr,-1)*( np.roll(Phi,-2) - np.roll(Phi,-1)  )
    - (1+kappa)/4*np.roll(rrr,-1)*(  np.roll(Phi,-1) - Phi  )
    return (PhihalfMinus, PhihalfPlus)


    
# ------------------------------------------------------------------------------------
###                   ••••• Advanced FE options. •••••                     ###
# ------------------------------------------------------------------------------------

def FE_NL_K(c,Phi):## not functional yet.
    '''Forwards euler nonlinear, high order, Muscl kappa scheme.
    advised for nonlinear problems
    inputs:
    c is the Courant number
    Phi is the dependent variable that is advected.'''
    # note I have not calculated the solution to the reimann problem or done flux splitting.
    L,R  = FCT_MUSCL_limited_NL(Phi)
    return Phi - c*( Riem(L,R) - np.roll(Riem(L,R),1) )
# ------------------------------------------------------------------------------------
###                   ••••• Advanced explicit RK options •••••                     ###
# ------------------------------------------------------------------------------------
    
def SSP33_NL_K(c,Phi):
    return 1/3*Phi+2/3*FE_NL_K(c,3/4*Phi+1/4*FE_NL_K(c,FE_NL_K(c,Phi))) 
    
    
    
def RK4Spectral_NL(Phi,dt,k,a):## this scheme should add in viscosity.
    dt = dt
    k1 =  dt*ft.ifft(a*k*1j*ft.fft( -FF(Phi) ))  
    #print(k1)
    k2 = np.real( dt*ft.ifft(a*k*1j*ft.fft( -FF(Phi+0.5*k1) )) )
    k3 = np.real( dt*ft.ifft(a*k*1j*ft.fft( -FF(Phi+0.5*k2) )) )   
    k4 = np.real( dt*ft.ifft(a*k*1j*ft.fft( -FF(Phi+k3) )) )      
       #FE in fourier space    
    Phi = Phi + 1/6*( k1 + 2*k2 + 2*k3 + k4 ) 
    return Phi    
        
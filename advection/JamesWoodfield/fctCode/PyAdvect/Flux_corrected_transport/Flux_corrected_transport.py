#!/usr/local/bin/python3 
from BaseSchemes_Linear_advection import *
"""
@author: jmw418
# -*- coding: utf-8 -*-
## i will work on the derivations i have done for muscle type schemes
"""
###Packages###
import numpy as np
import numpy as np
import scipy.linalg
from numpy import linalg 
import matplotlib.pyplot as plt
import time
from time import process_time 

# ------------------------------------------------------------------------------------
###                          ••••• BorisandBook •••••                            ###
# ------------------------------------------------------------------------------------
def BorisandBook(c,Phi):
    '''Book, Boris, Hain. 1973, 1974'''
    nx = len(Phi);
    Phinew = FELL(c,Phi)
    
    ### Second pass  ###
    AF = np.zeros(nx); AFC = np.zeros(nx);
    ## Antidiffusive flux correction at j+0.5
    
    AF = c*(MUSCL_linear(Phi) - Phi ) 
    
    
    ### prelimiting step.
    for j in range(0,nx):
        if ( AF[j%nx]*(Phinew[(j+1)%nx] - Phinew[(j)%nx]) < 0):
            if ( AF[j%nx]*(Phinew[(j+2)%nx] - Phinew[(j+1)%nx]) < 0):
                AF[j%nx] = 0
            if ( AF[j%nx]*(Phinew[(j)%nx] - Phinew[(j-1)%nx]) < 0):
                AF[j%nx] = 0
                
                
    ## Boris and Book modification, to the flux
    for i in range(0,nx):
        AFC[i] = np.sign(AF[i])*max(0,min(abs(AF[i]),np.sign(AF[i])*(Phinew[(i+2)%nx] - Phinew[(i+1)%nx]),  \
        np.sign(AF[i])*(Phinew[i] - Phinew[(i-1)%nx]) )) 
    
    return Phinew - ( AFC - np.roll(AFC,1) )
    
def BorisandBook_theta(c,Phi):
    nx = len(Phi);
    ### First pass  ### I choose the first pass to be crank nicholson, or weighted.
    if c<=2:
        Phinew = ThetaLL(c,Phi,0.5)
    if c>2:
        Phinew = ThetaLL(c,Phi,1-1/c)
    
    ### Second pass  ###
    AF = np.zeros(nx); AFC = np.zeros(nx);
    ## Antidiffusive flux correction at j+0.5
    AF = c*(MUSCL_linear(Phi) - Phi) 
    
    ### prelimiting step. # probably a little unnessary.
    for j in range(0,nx):
        if ( AF[j%nx]*(Phinew[(j+1)%nx] - Phinew[(j)%nx]) < 0):
            if ( AF[j%nx]*(Phinew[(j+2)%nx] - Phinew[(j+1)%nx]) < 0):
                AF[j%nx] = 0
            if ( AF[j%nx]*(Phinew[(j)%nx] - Phinew[(j-1)%nx]) < 0):
                AF[j%nx] = 0
                
    ## Boris and Book modification, to the flux
    
    for i in range(0,nx):
        AFC[i] = np.sign(AF[i])*max(0,min(abs(AF[i]),np.sign(AF[i])*(Phinew[(i+2)%nx] - Phinew[(i+1)%nx]),  \
        np.sign(AF[i])*(Phinew[i] - Phinew[(i-1)%nx]) )) 
    #print(AFC)
    return Phinew - ( AFC - np.roll(AFC,1) )
    
def BorisandBook_BE(c,Phi):
    '''this algorithm seems more stable than zalesac'''
    nx = len(Phi);
    Phinew = np.zeros(nx);

    ### First pass  ### I choose the first pass to be 
    
    Phinew = BELL(c,Phi)
    
    ### Second pass  ###
    AF = np.zeros(nx); AFC = np.zeros(nx);
    ## Antidiffusive flux correction at j+0.5
    AF = c*(MUSCL_linear(Phi) - Phinew) 
    
    
    ### prelimiting step.
    for j in range(0,nx):
        if ( AF[j%nx]*(Phinew[(j+1)%nx] - Phinew[(j)%nx]) < 0):
            if ( AF[j%nx]*(Phinew[(j+2)%nx] - Phinew[(j+1)%nx]) < 0):
                AF[j%nx] = 0
            if ( AF[j%nx]*(Phinew[(j)%nx] - Phinew[(j-1)%nx]) < 0):
                AF[j%nx] = 0
                
    ## Boris and Book modification, to the flux
    for i in range(0,nx):
        AFC[i] = np.sign(AF[i])*max(0,min(abs(AF[i]),np.sign(AF[i])*(Phinew[(i+2)%nx] - Phinew[(i+1)%nx]),  \
        np.sign(AF[i])*(Phinew[i] - Phinew[(i-1)%nx]) )) 
    
    return Phinew - ( AFC - np.roll(AFC,1) )
    
    
def Zalesak(c,Phi):
    nx = len(Phi)
    ### First pass  ###
    
    Phinew = FELL(c,Phi)
    
    ### Second pass  ###
    AF = np.zeros(nx);AFC = np.zeros(nx);C = np.zeros(nx);
    PP = np.zeros(nx);QP = np.zeros(nx);RP = np.zeros(nx);
    PM = np.zeros(nx);QM = np.zeros(nx);RM = np.zeros(nx);
    
    ## Antidiffusive flux correction at j+0.5
    AF = c*(MUSCL_linear(Phi) - Phi)
    
    # FCT_FEWENO5_L(Phi)  # FCT_MUSCL_limited(Phi), FCT_MUSCL_linear(Phi)
    
    maxim = np.maximum(Phi, Phinew)
    minim = np.minimum(Phi, Phinew)
    maxim = np.maximum(maxim, np.maximum(np.roll(maxim,-1), np.roll(maxim,1)))
    minim = np.minimum(minim, np.minimum(np.roll(minim,-1), np.roll(minim,1)))
    
    ### prelimiting step.
    for j in range(0,nx):
        if ( AF[j%nx]*(Phinew[(j+1)%nx] - Phinew[(j)%nx]) < 0):
            if ( AF[j%nx]*(Phinew[(j+2)%nx] - Phinew[(j+1)%nx]) < 0):
                AF[j%nx] = 0
            if ( AF[j%nx]*(Phinew[(j)%nx] - Phinew[(j-1)%nx]) < 0):
                AF[j%nx] = 0
    ### Zalesak limiter
    PP = np.maximum(0,np.roll(AF,1)) - np.minimum(AF,0) ## flux in
    PM = np.maximum(AF,0) - np.minimum(0,np.roll(AF,1)) ## flux out
    QP = (maxim - Phinew)*c
    QM = (Phinew - minim)*c
    
    for j in range(0,nx):
        if PP[j]>0:
            RP[j] = min(1,QP[j]/PP[j])  
    
    for j in range(0,nx):
        if PM[j]>0:
            RM[j] = min(1,QM[j]/PM[j]) 
    
    for j in range(0,nx):
        if (AF[j]>=0):
            C[j] = min(RP[(j+1)%nx],RM[(j)%nx])
    for j in range(0,nx):
        if (AF[j]<0):
            C[j] = min(RP[(j)%nx],RM[(j+1)%nx])
            
    for j in range(0,nx):
        AFC[j] = AF[j]*C[j]
    
    return Phinew -  ( AFC - np.roll(AFC,1) )
    
def MUSCL_FE(c,Phi):
    Phinew = FELL(c,Phi)
    
    ## Antidiffusive flux correction at j+0.5
    AF = c*(MUSCL_linear(Phi) - Phi)

    return Phinew -  ( AF - np.roll(AF,1) )


def Zalesak_BE(c,Phi):
    nx = len(Phi)
    ### First pass  ### 
    Phinew = BELL(c,Phi)

    ### Second pass  ###
    AF = np.zeros(nx);AFC = np.zeros(nx);C = np.zeros(nx);
    PP = np.zeros(nx);QP = np.zeros(nx);RP = np.zeros(nx);
    PM = np.zeros(nx);QM = np.zeros(nx);RM = np.zeros(nx);

    ## Antidiffusive flux correction at j+0.5
    AF = ( MUSCL_linear(Phi) - Phi)*c ### np.random() 
    
    ## this correction is explicit so must obey a CFL number to be linearly stable. 0.5?
    
    # FCT_FEWENO5_L(Phi)  # FCT_MUSCL_limited(Phi), FCT_MUSCL_linear(Phi)
    maxim = np.maximum(Phi, Phinew)
    minim = np.minimum(Phi, Phinew)
    maxim = np.maximum(maxim, np.maximum(np.roll(maxim,-1), np.roll(maxim,1)))
    minim = np.minimum(minim, np.minimum(np.roll(minim,-1), np.roll(minim,1)))

    ### prelimiting step.
    for j in range(0,nx):
        if ( AF[j%nx]*(Phinew[(j+1)%nx] - Phinew[(j)%nx]) < 0):
            if ( AF[j%nx]*(Phinew[(j+2)%nx] - Phinew[(j+1)%nx]) < 0):
                AF[j%nx] = 0
            if ( AF[j%nx]*(Phinew[(j)%nx] - Phinew[(j-1)%nx]) < 0):
                AF[j%nx] = 0


    PP = (np.maximum(0,np.roll(AF,1)) - np.minimum(AF,0))/(c)
    PM = (np.maximum(AF,0) - np.minimum(0,np.roll(AF,1)))/(c)
    #print("PP,PM",PP,PM)
    QP = (maxim - Phinew) ## we can only correct the explicit part by this much stably 
    QM = (Phinew - minim)
    #print("QP,QM",QP,QM)
    
    
    ## the limiting here ensures that higher order is done
    for j in range(0,nx):
        if PP[j]>0:
            RP[j] = np.minimum(1,QP[j]/PP[j])  # introduce ,QP[j]/PP[j]*1/c**2? however just recovers first order. 
    for j in range(0,nx):
        if PM[j]>0:
            RM[j] = np.minimum(1,QM[j]/PM[j]) # introduce ,QP[j]/PP[j]*1/c**2
    #print("RP,RM",RP,RM)
    
    for j in range(0,nx):
        if (AF[j]>=0):
            C[j] = np.minimum(RP[(j+1)%nx],RM[(j)%nx])
    for j in range(0,nx):
        if (AF[j]<0):
            C[j] = np.minimum(RP[(j)%nx],RM[(j+1)%nx])
    ## issue is that this is outputting 1, how is it doing this
    for j in range(0,nx):
        AFC[j] = AF[j]*C[j]
    #print("C,AFC",C,AFC)
    #plt.clf()
    #plt.plot(C)
    #plt.draw()
    #plt.show()
    B = np.identity(nx)
    #print(C)
    gamma = 0
    for j in range(0,nx):
        B[j,j] = 1  + gamma*AFC[j];
        B[j,(j-1)%nx] = - gamma*AFC[(j-1)%nx] ;

    Phinew = Phinew - (1-gamma)*( AFC - np.roll(AFC,1) )
    #Phinew = scipy.linalg.solve(B,Phinew)
    return Phinew
#
#
#
#
#
#
def MUSCL_theta(c,Phi):
    if c<1:
        Phi = CNMUSCL_L(c,Phi)
    if c>=1:
        Phi = ThetaMUSCL_L(c,Phi,(1 - 0.5/c) )
    return Phi
#
def Zalesak_theta(c,Phi):
    '''this algorithm seems more stable than zalesac'''
    nx = len(Phi);
    Phinew = np.zeros(nx);

    ### First pass  ### I choose the first pass to be 
    if c<=2:
        Phinew = ThetaLL(c,Phi,0.5)
    if c>2:
        Phinew = ThetaLL(c,Phi,1-1/c)
    

    ### Second pass  ###
    AF = np.zeros(nx);AFC = np.zeros(nx);C = np.zeros(nx);
    PP = np.zeros(nx);QP = np.zeros(nx);RP = np.zeros(nx);
    PM = np.zeros(nx);QM = np.zeros(nx);RM = np.zeros(nx);

    ## Antidiffusive flux correction at j+0.5
    AF = ( MUSCL_linear(Phi) - Phi)*c
    
    #i think that it is Phi, but this doesnt work, replacing with Phinew seems to work? 
    # FCT_FEWENO5_L(Phi)  # FCT_MUSCL_limited(Phi), FCT_MUSCL_linear(Phi)

    maxim = np.maximum(Phi, Phinew)
    minim = np.minimum(Phi, Phinew)
    maxim = np.maximum(maxim, np.maximum(np.roll(maxim,-1), np.roll(maxim,1)))
    minim = np.minimum(minim, np.minimum(np.roll(minim,-1), np.roll(minim,1)))

    # ### prelimiting step.

    ### prelimiting step.
    for j in range(0,nx):
        if ( AF[j%nx]*(Phinew[(j+1)%nx] - Phinew[(j)%nx]) < 0):
            if ( AF[j%nx]*(Phinew[(j+2)%nx] - Phinew[(j+1)%nx]) < 0):
                AF[j%nx] = 0
            if ( AF[j%nx]*(Phinew[(j)%nx] - Phinew[(j-1)%nx]) < 0):
                AF[j%nx] = 0
    ### Zalesak limiter
    PP = (np.maximum(0,np.roll(AF,1)) - np.minimum(AF,0))/(c)
    PM = (np.maximum(AF,0) - np.minimum(0,np.roll(AF,1)))/(c)
    
    QP = (maxim - Phinew) ## we can only correct the explicit part by this much stably 
    QM = (Phinew - minim)
    for j in range(0,nx):
        if PP[j]>0:
            RP[j] = min(1,QP[j]/PP[j])  # ,QP[j]/(PP[j]*c*c)
    for j in range(0,nx):
        if PM[j]>0:
            RM[j] = min(1,QM[j]/PM[j]) # ,QM[j]/(PM[j]*c*c)

    for j in range(0,nx):
        if (AF[j]>=0):
            C[j] = min(RP[(j+1)%nx],RM[(j)%nx])
    for j in range(0,nx):
        if (AF[j]<0):
            C[j] = min(RP[(j)%nx],RM[(j+1)%nx])

    for j in range(0,nx):
        AFC[j] = AF[j]*C[j]
    #plt.clf()
    #plt.plot(C)
    #plt.draw()
    #plt.show()
    B = np.identity(nx)
    #print(C)
    gamma = 0
    for j in range(0,nx):
        B[j,j] = 1  + gamma*AFC[j];
        B[j,(j-1)%nx] = - gamma*AFC[(j-1)%nx] ;

    Phinew = Phinew - (1-gamma)*( AFC - np.roll(AFC,1) )
    #Phinew = scipy.linalg.solve(B,Phinew)
    return Phinew
    
    
    
    
def MUSCL_FE(c,Phi):
    Phinew = FELL(c,Phi)

    ## Antidiffusive flux correction at j+0.5
    AF = c*(MUSCL_linear(Phi) - Phi)

    return Phinew -  ( AF - np.roll(AF,1) )


def MUSCL_BE(c,Phi):
    Phinew = BELL(c,Phi)
    AF = c*(MUSCL_linear(Phi) - Phinew)
    return Phinew -  ( AF - np.roll(AF,1) )


def Zalesak_BEHilary(c,Phi):
    nx = len(Phi)
    ### First pass  ###

    Phinew = BELL(c,Phi)

    ### Second pass  ###
    AF = np.zeros(nx);AFC = np.zeros(nx);C = np.zeros(nx);
    PP = np.zeros(nx);QP = np.zeros(nx);RP = np.zeros(nx);
    PM = np.zeros(nx);QM = np.zeros(nx);RM = np.zeros(nx);

    ## Antidiffusive flux correction at j+0.5
    AF = c*(MUSCL_linear(Phi) - Phinew)

    maxim = np.maximum(Phi, Phinew)
    minim = np.minimum(Phi, Phinew)
    maxim = np.maximum(maxim, np.maximum(np.roll(maxim,-1), np.roll(maxim,1)))
    minim = np.minimum(minim, np.minimum(np.roll(minim,-1), np.roll(minim,1)))

    ### prelimiting step.
    for j in range(0,nx):
        if ( AF[j%nx]*(Phinew[(j+1)%nx] - Phinew[(j)%nx]) < 0):
            if ( AF[j%nx]*(Phinew[(j+2)%nx] - Phinew[(j+1)%nx]) < 0):
                AF[j%nx] = 0
            if ( AF[j%nx]*(Phinew[(j)%nx] - Phinew[(j-1)%nx]) < 0):
                AF[j%nx] = 0

    ### Zalesak limiter
    PP = np.maximum(0,np.roll(AF,1)) - np.minimum(AF,0) ## flux in
    PM = np.maximum(AF,0) - np.minimum(0,np.roll(AF,1)) ## flux out
    QP = (maxim - Phinew)*min(c,1)
    QM = (Phinew - minim)*min(c,1)

    for j in range(0,nx):
        if PP[j]>0:
            RP[j] = min(1,QP[j]/PP[j])  

    for j in range(0,nx):
        if PM[j]>0:
            RM[j] = min(1,QM[j]/PM[j]) 

    for j in range(0,nx):
        if (AF[j]>=0):
            C[j] = min(RP[(j+1)%nx],RM[(j)%nx])
    for j in range(0,nx):
        if (AF[j]<0):
            C[j] = min(RP[(j)%nx],RM[(j+1)%nx])

    for j in range(0,nx):
        AFC[j] = AF[j]*C[j]

    return Phinew -  ( AFC - np.roll(AFC,1) )



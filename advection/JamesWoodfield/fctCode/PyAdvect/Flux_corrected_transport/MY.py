#!/usr/local/bin/python3 

from Flux_corrected_transport import *
from BaseSchemes_Linear_advection import *
from BaseSchemes_NonLinear_advection import *
"""
@author: jmw418
# -*- coding: utf-8 -*-
"""
###Packages###
import numpy as np
import scipy.linalg
from numpy import linalg 
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
from matplotlib.collections import LineCollection
import os
import glob
import sys
import copy
import time
from time import process_time 
import random 
#from PIL import Imager
#xRimport natsort 

# initial condition
def IC(x):
    return 0.0 + np.exp(-100*(x-0.3)**2) + 1*(x<0.8)*(x>0.6)

    
def main():
    ## resolution of saved images.
    res = 300
    ###***    initialisation of constants  ***###
    nx = 100;nt = 240;a = 0.12;tstart = 0;tend = 8;xmin=0.;xmax = 1;
    #nx = 20;nt = 1 ;a = 12;tstart = 0;tend = 0.01;xmin=0.;xmax = 1;
    ###••••• derived parameters •••••###
    dx = (xmax-xmin)/nx; dt = (tend-tstart)/nt;
    c = a*nx/nt*(tend-tstart)/(xmax-xmin);
    
    ###••••• initialisation of structures •••••###
    x = np.arange(xmin,xmax,dx);
    time = np.linspace(tstart,tend,nt+1);
    Phi = np.zeros([nx]); 
    #print("c;",c)
    #print(nx,dx,x)
    
    #initialisation of fourier structure
    #nx = nx+1 
    if ((nx+1)%2==0):
        print("need odd number of space points")
    delta_k = 2*np.pi/(nx*dx)
    k1 =  np.arange(0,delta_k*(nx/2),delta_k)
    k2 =  np.arange(-delta_k*(nx+1)/2,0,delta_k)
    kkr= np.concatenate((k1,k2))
    #snx = nx-1
    
    ###***   initial condition ***###
    Phi = IC(x); 
    
    print("initial Condition Set")
    limiter = 'van_L'
    ###****  Storing quantities  ****###
    Store = np.zeros([nt,nx])
    tv = np.zeros(nt+1);err1 = np.zeros(nt+1);err2 = np.zeros(nt+1);
    mass = np.zeros(nt+1);energy = np.zeros(nt+1);variance = np.zeros(nt+1);
    
    
    ###***  Numerical Schemes ***###RK4Spectral_NL
    solvers = ['FELL','MUSCL_FE', 'Zalesak']
    '''the above implements Zalesak_backward euler and the Lee linearisation of the Van leer-k slope limiting'''
    
    ''' the correction is done explicitly for Zalasak, the limiting doesnt suppress the fact that this is an unstable scheme '''
    #['Zalesak_BE','BEMUSCL_L','BELL','BorisandBook_BE'] # backward euler. 
    #['Zalesak_theta','BorisandBook_theta','MUSCL_theta'] these arent really methods 
    #['SSPRK3_4stage','SSP33_L'] 
    #['Zalesak_BE','BEMUSCL_L','BELL','BorisandBook_BE']
    #'BorisandBook_theta','Zalesak_theta'
    #['ZalesakBE','BEMUSCL_L','BELL','BorisandBook']
    #['RK4Spectral_L','SSP33WENO5FCT','FEMUSCL_LW_2']
    #['Zalesak','FELL','FEMUSCL_L','ZalesakBE','BELL','SSP33WENO5FCT']
    #['ZalesakBE','BELL']
        
        
    u_solutions = np.zeros((len(solvers),len(time),len(x)))
    
    ###***   Loop for solvers ***### 
    
    for k in range(len(solvers)):
        print(solvers[k])
        Phi = IC(x) 
        un = np.zeros((len(time), len(x)))
        ###***   Loop for time ***###
        t1_start = process_time() 
        for i in range(0,nt):
            tv[i] = Total_variation(Phi)
            #mass[i],energy[i],variance[i] = Measures(Phi,x,i,a,dt)
            #Animate(Phi,x,i,a,dt,tv)
            #AnimationSaver(Phi,x,i,a,dt,tv,c,scheme)
            Store[i,:] = Phi[:]
        
        
            ###••••• Loop for Space•••••###
            x1_start = process_time()
            ###•Linear advection schemes••###
            
            ###•simple schemes••###
            if( solvers[k]=='BELL' ):
                Phi =  BELL(c,Phi)
            if( solvers[k]=='BELL_fast' ):
                Phi =  BELL_fast(c,Phi)
            if( solvers[k]=='FELL' ):
                Phi =  FELL(c,Phi)
                
            ###•Muscl schemes••###
            if( solvers[k]=='FEMUSCL_L' ):
                Phi =  FEMUSCL_L(c,Phi)
            if( solvers[k]=='FEMUSCL_LW' ):
                Phi =  FEMUSCL_LW(c,Phi)
            if( solvers[k]=='BEMUSCL_L' ):
                Phi =  BEMUSCL_L(c,Phi)
                
            ## WENO5
            if( solvers[k]=='FEWENO5_L' ): 
                Phi = FEWENO5_L(c,Phi)
            if( solvers[k]=='SSP33WENO5_L' ): 
                Phi = SSP33WENO5_L(c,Phi)
            if( solvers[k]=='SSP104WENO5_L' ): 
                Phi = SSP104WENO5_L(c,Phi)
                
            if( solvers[k]=='SSP104ENO3_L' ): 
                Phi = SSP104ENO3_L(c,Phi)
                
            ###• FCT schemes •###
            if( solvers[k]=='BorisandBook' ):
                Phi =  BorisandBook(c,Phi)
            if( solvers[k]=='BorisandBook_theta' ):
                Phi =  BorisandBook_theta(c,Phi)
            if( solvers[k]=='BorisandBook_BE' ):
                Phi =  BorisandBook_BE(c,Phi)
                
            if( solvers[k]=='Zalesak' ):
                Phi =  Zalesak(c,Phi)
            if( solvers[k]=='ZalesakLW' ):
                Phi =  ZalesakLW(c,Phi)
            if( solvers[k]=='Zalesak_BE' ):
                Phi =  Zalesak_BE(c,Phi)
            if( solvers[k]=='Zalesak_theta' ):
                Phi =  Zalesak_theta(c,Phi)    
            if( solvers[k]=='Zalesak_BEHilary' ):
                Phi =  Zalesak_BEHilary(c,Phi)
                
                
            if( solvers[k]=='MUSCL_theta' ):
                Phi =  MUSCL_theta(c,Phi)
            if( solvers[k]=='MUSCL_FE' ):
                Phi =  MUSCL_FE(c,Phi)
            if( solvers[k]=='MUSCL_BE' ):
                Phi =  MUSCL_BE(c,Phi)
                
            ### arbitrary ordered stencils
            if( solvers[k]=='FE_centered_ho' ):
                Phi = FE_centered_ho(c,Phi,5)
            if( solvers[k]=='FE_upwind_ho' ):
                Phi = FE_upwind_ho(c,Phi,3)   
            ###
            
            
            
                
            if( solvers[k]=='SSP104MUSCL_L' ): 
                Phi = SSP104MUSCL_L(c,Phi)
                
            if( solvers[k]=='SSPRK3_4stage_C' ): 
                Phi = SSPRK3_4stage_C(c,Phi) 
            if( solvers[k]=='SSPRK3_4stage' ): 
                Phi = SSPRK3_4stage(c,Phi)
            
            
            if( solvers[k]=='FEMUSCL_LW_2' ):
                Phi =  FEMUSCL_LW_2(c,Phi)
                
            if( solvers[k]=='SSP33WENO5FCT' ): 
                Phi = SSP33WENO5FCT(c,Phi)
                
                
            if( solvers[k]=='SSP33_L' ):
                Phi = SSP33_L(c,Phi)
                
            if( solvers[k]=='TRBDF2MUSCL_L' ):
                Phi = TRBDF2MUSCL_L(c,Phi)
                
            ### adaptive timesteppers ###
            # schemes implement substages based on cfl, to be robust.
            if( solvers[k]=='SSPIRK2_2s_L' ):
                Phi = SSPIRK2_2s_L(c,Phi)
                
            ###•spectral••###
            if( solvers[k]=='RK4Spectral_L' ):
                ## only working for odd nx
                Phi = RK4Spectral_L(Phi,dt,kkr,a)
            if( solvers[k]=='EXP_Spectral_L' ):
                ## only working for odd nx
                Phi = EXP_Spectral_L(Phi,dt,kkr,a)
            
            
            
            ###•NonLinear hyperbolic schemes••###
            
            if( solvers[k]=='FE_NL_L' ):
                Phi = FE_NL_L(c,Phi)
            if( solvers[k]=='FE_NL_K' ):
                Phi = FE_NL_K(c,Phi)
            if( solvers[k]=='SSP33_NL_K' ):
                Phi = SSP33_NL_K(c,Phi)
            if( solvers[k]=='BE_NL_L' ):
                Phi = BE_NL_L(c,Phi)
            if( solvers[k]=='CN_NL_L' ):
                Phi = BE_NL_L(0.5*c,FE_NL_L(0.5*c,Phi))
                
                
            if( solvers[k]=='RK4Spectral_NL' ):
                Phi = RK4Spectral_NL(Phi,dt,kkr,a)
                
            x1_end = process_time()
            un[i,:] = Phi[:]   
             
        t1_stop = process_time()
        u_solutions[k,:,:] = un  
        
        print('----------------------------------')
        print('  SCHEME  : ',solvers[k])
        print('  Time step  : ',dt)
        print('  CFL        : ',c)
        print( '  Nb points  : ',nx)
        print('  Simulation steps : ',nt)
        print('  Elapsed time for 1 step : ', x1_end-x1_start)  
        print('  Elapsed time during the whole program in seconds: ',t1_stop-t1_start) 
        print('  Elapsed time for whole program, not functional: ') 
        print('----------------------------------')
        
    # ### below is the plot of the total variation.
#         plt.ylabel('y')
#         plt.xlabel('time')
#         plt.plot(time,tv,linewidth = 0.8,label = '.'+str(solvers[k])+'.')
#         plt.title('Total Variation')
#         plt.legend()
#     plt.savefig('TV.png',dpi = res)
        
        
        
        
    # the below does the multiple plot animations     #    #
    for i in range(0,nt):
        plt.clf()
        for k in range(len(solvers)):
            TOTAL_plot(x,solvers[k],time,c,u_solutions[k,i,:])
        plt.plot(x,IC((x-i*a*dt)%1 ),'k',label = 'Analytic')
        plt.draw()
        plt.pause(0.003)
    plt.clf()
    #the above does multiple plot animations
    
    
    
    
    
    
    
    # below is final time plot
    solversString = ''
    for k in range(len(solvers)):
        TOTAL_plot(x,solvers[k],time,c,u_solutions[k,nt-1,:])
        solversString = solversString + str(solvers[k]) + '_'
    plt.plot(x,IC((x-i*a*dt)%1 ),'k',label = 'Analytic')
    plt.savefig(solversString + 'Final_time_'+str(nx)+'_'+str(nt)+'_'+str(limiter)+'.pdf')
    plt.show()
    #Plotting_TV(tv,time) plot the total variation
    #Plotting_Error(err,time)
    
    
    
    
    
    
    
# ------------------------------------------------------------------------------------
###                          ••••• Plotting tools •••••                            ###
# ------------------------------------------------------------------------------------
def TOTAL_plot(x,scheme,time,c,Phi):
    '''Plots multiple at once'''
    nx = len(Phi); nt = len(time); tstart = min(time); tend = max(time);
    xmin = min(x); xmax = max(x); dx = (xmax-xmin)/(nx); dt = (tend-tstart)/(nt); 
    plt.axis([0, 1, -0.1, 1.3])
    plt.xlabel('x')
    plt.ylabel('$\phi$')
    plt.title(' Tend='+str(tend)+'; cfl='+str(c)+'; (nx,nt)='+str((nx,nt))+';')
    plt.grid(True)
    plt.plot(x,Phi,label = scheme)
    plt.legend()
    return
    
def Animate(Phi,x,i,a,dt,tv):
    plt.clf()
    plt.title(tv[i])
    plt.axis([0, 1, -0.1, 1.3])
    plt.plot(x,Phi,'b')
    plt.plot(x,IC((x-i*a*dt)%1 ),'r')
    plt.draw()
    plt.pause(0.003)
    return
 

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

    levels = np.linspace(0-10e-7,1.0+10e-17,41)
    pcm = plt.contourf(X, T, Y,levels = levels,cmap = cmap,extend = "both")
    
    cmap = copy.copy(plt.cm.get_cmap("hot"))
    pcm.cmap.set_over("magenta")
    pcm.cmap.set_under('cyan')
    fig.colorbar(pcm)

    plt.ylabel('Time')
    plt.xlabel('Space')
    levels=[-0.00000, 1.00000]
    #CS = plt.contour(X, T, Y, levels=levels,colors='k')
    
    #plt.clabel(CS,fontsize=7)
    plt.title(''+str(scheme)+' SpaceTimePlot at cfl '+str(c)+'')
    plt.savefig(''+str(scheme)+'-SpaceTimePlot.png',dpi = res)
    
    return 
    
def ICSaver(Phi,x,c,scheme,res):
    plt.clf()
    plt.title('Initial Conditions')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.plot(x,IC(x),'k',label = 'initial conditions')
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


    
    


if __name__ == "__main__":
    # execute only if run as a script
    #ComputationalTimeTests()
    main()
    

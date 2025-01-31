#! /usr/local/opt/python@3.8/bin/python3

"""prototype advection"""
import numpy as np
import matplotlib.pyplot as plt
#import pyamg
""""@MISC{OlSc2018,
      author = "Olson, L. N. and Schroder, J. B.",
      title = "{PyAMG}: Algebraic Multigrid Solvers in {Python} v4.0",
      year = "2018",
      url = "https://github.com/pyamg/pyamg",
      note = "Release 4.0"
      }"""
#from sympy import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.linalg as sla
import scipy.sparse.linalg
from scipy import sparse
from time import process_time
from scipy.sparse import dok_matrix
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import spilu# ------------------------------------------------------------------------------------

def __main__():

# domain
    xmin = 0.; xmax = 1.;
    tmax = 1;
# discretisation
    nx = 10001; # number of points including the edges.
    nt = 1000; 
    a = 0.9;
#derived parameters
    dx = (xmax - xmin) /(nx-1)
    dt = tmax/(nt)
    c_1 = dt/dx*a; 
    print('cfl=',c_1)
    xc = np.linspace(xmin+0.5*dx,xmax-0.5*dx,nx-1)# the vector of the cell centers
    xf = np.linspace(xmin,xmax-dx,nx-1)# the vector of the faces with first removed # 
    Ufx = InitialVelocity(xf) 
    Phi = initalCondition(xc) # define scalar at centers. 
    Ufx = Ufx*c_1
    N = len(Phi)
    Phi2 = initalCondition(xc)
    for n in range(0,nt):
        t1_start = process_time() 
        Phi = diaMatrix(Ufx,Phi) 
        Phi2 = BEMUSCL_now(Ufx,Phi2)
        if n%100 ==0:
            plt.clf()
            plt.plot(xc,Phi,'k')
            plt.plot(xc,Phi2,'g')
            plt.draw()
            plt.pause(0.001)
    plt.show()
        
    
def initalCondition(xxc):
    return 0 + np.exp(-100*(xxc-0.75)**2) + 1*(xxc>0.15)*(xxc<0.45)
    
def InitialVelocity(a):
    "input a, the (x) coordinate for a face, or quadrature points, in which you want velocity."
    """currently only for constant sign"""
    U = 1 + 0*a#0.3*np.sin(a)
    return U
def DV(a,b):
    ''' Finally a safe divide function'''
    return a/( b + 1.0e-12*(b==0)+ 1.0e-16  )
def pos(c):
    return 0.5*(c+np.abs(c))
def neg(c):
    return 0.5*(c-np.abs(c))
def upstream_flux(m,p,c):
    """m is the j, p is the j+1"""
    return p*neg(c) + m*pos(c) 

def FL(r,glimiter):
    """some flux limiters"""
    if (glimiter == 'minmod'):
        r = np.maximum(np.minimum(r,1),0)
    if (glimiter == 'superbee'):
        r = np.maximum(0,np.minimum(2*r,1),np.minimum(r,2))
    if (glimiter == 'vanleer'):
        r =  DV( r+np.abs(r) , 1+np.abs(r) )
    if (glimiter == 'korem'):
        r = np.maximum(np.minimum(2,np.minimum(1/3 + 2/3*r,2*r)),0)
    return r
    
    
def diaMatrix(Ufx,f):
    """ this is the fastest O(N) construction."""
    """constuction of the BE Donor cell algorithm"""
    N = len(f)
    main = 1 + pos(Ufx) - np.roll(neg(Ufx),1) 
    lower = -np.roll(pos(Ufx),1) 
    upper = neg(Ufx)
    
    lower = np.roll(lower,-1)
    S = dia_matrix( (N,N),dtype = float)
    S.setdiag(main, k=0)
    S.setdiag(upper, k=1)
    S.setdiag(lower, k=-1)
    S.setdiag(upper[-1], k=-(N-1))
    S.setdiag(lower[-1], k=+(N-1))
    f,q= scipy.sparse.linalg.gmres(S,f)
    return f
    

def BEMUSCL_now(Ufx,f):
    '''MUSCL '''
    N= len(f)
    t1_start = process_time()
    kappa = 1/3 ## Kappa scheme is standard flux limiter if 0, and 3rd order accurate for 1/3
    glimiter = 'superbee'
    r =  DV( f - np.roll(f,1) , np.roll(f,-1) - f ) ## ratio of successive grads
    rr = DV(1,r)  ## this is the usual one.
    r = FL(r,glimiter)
    rr = FL(rr,glimiter)
    main = 1 + Ufx + Ufx*(1-kappa)/4*rr - Ufx*(1+kappa)/4*r - Ufx*(1+kappa)/4*np.roll(r,1);
    upper = Ufx*(1+kappa)/4*r;
    lower = - Ufx*(1-kappa)/4*rr -Ufx - Ufx*(1-kappa)/4*np.roll(rr,1) + Ufx*(1+kappa)/4*np.roll(r,1);
    lower2 = Ufx*(1-kappa)/4*np.roll(rr,1);
    S = dia_matrix( (N,N),dtype = float)
    
    lower = np.roll(lower,-1)
    lower2 = np.roll(lower2,-2)
    # Because We have to roll into the correct positions
    
    S.setdiag(main, k=0)
    S.setdiag(upper, k=1)
    S.setdiag(upper[-1], k=-(N-1))
    
    S.setdiag(lower, k=-1)
    S.setdiag(lower[-1], k=+(N-1))
    S.setdiag(lower2, k=-2)
    S.setdiag(lower2[N-2:N], k=+(N-2))
    
    t1_stop = process_time()
    print("build=",- t1_start +t1_stop)
    "https://relate.cs.illinois.edu/course/CS556-f16/file-version/38da5c7d14e22927a994eeba759f0974b2169fb2/demos/ILU%20preconditioning.html"
    t1_start = process_time()
    S = sparse.csc_matrix(S)
    B = spilu(S)
    Mz = lambda r: B.solve(r)
    Minv = scipy.sparse.linalg.LinearOperator(S.shape, Mz)
    #f,q = pyamg.krylov.gmres(S, f, x0=f, tol=1e-30, restrt=20, maxiter=100, M=Minv)
    f = scipy.sparse.linalg.spsolve(S,f,use_umfpack=True)
    t1_stop = process_time()
    print("Solve_time",- t1_start +t1_stop)
    return f
    
    
    
    
    
if __name__ == "__main__":
    __main__()

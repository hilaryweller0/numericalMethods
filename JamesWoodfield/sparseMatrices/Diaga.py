#! /usr/local/bin/python3 
import numpy as np
import matplotlib.pyplot as plt
#from sympy import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.linalg as sla
import scipy.sparse.linalg
from scipy import sparse
from time import process_time

from scipy.sparse import dia_matrix
# ------------------------------------------------------------------------------------


def __main__():

# domain
    xmin = 0.; xmax = 1.;
    ymin = 0.; ymax = 1.; 
    tmax = 10;
# discretisation
    nx = 100; ny = 100; # number of points including the edges.
    nt = 100; 
    a = 0.5;
#derived parameters
    dx = (xmax - xmin) /(nx-1)
    dy = (ymax - ymin) /(ny-1) 
    dt = tmax/(nt)
    c_1 = dt/dx*a; c_2 = dt/dy*a; 
    
    xc = np.linspace(xmin+0.5*dx,xmax-0.5*dx,nx-1)# the vector of the cell centers
    yc = np.linspace(ymin+0.5*dy,ymax-0.5*dy,ny-1)
    xf = np.linspace(xmin,xmax-dx,nx-1)# the vector of the faces with first removed # 
    yf = np.linspace(ymin,ymax-dy,ny-1)# this is because we use periodic BC's
    
    xv, yv   = np.meshgrid(xf, yf, sparse=False, indexing='xy') #the vertices. 
    xxf, xyf = np.meshgrid(xf, yc, sparse=False, indexing='xy') #the x faces positions 
    yxf, yyf = np.meshgrid(xc, yf, sparse=False, indexing='xy') #the y faces positions 
    xxc, yyc = np.meshgrid(xc, yc, sparse=False, indexing='xy') #the c enters
    ## the xy indexing, does have some j,i to access the elements?

    Ufx,Vfx = InitialVelocity(xxf,xyf) # velocity vectors at x face positions
    Ufy,Vfy = InitialVelocity(yxf,yyf) # velocity vectors at y face positions
    
    #quiverplot(xxf,xyf,yxf,yyf,Ufx,Vfx,Ufy,Vfy)
    
    Phi = initalCondition(xxc,yyc) # define scalar at centers. 
    f = Phi
    nx = len(f[0,:]); ny = len(f[:,0]); #print("nx,ny=",nx,ny);
    f = f.reshape(-1); N = len(f); # 
    Ufx = Ufx.reshape(-1)*c_1
    Vfy = Vfy.reshape(-1)*c_2
    #plt.pcolormesh(Phi) #plt.show()
    A = DenseBuilder(f,Ufx,Vfy,nx,ny)
    fig,ax = plt.subplots()
    ax.spy(A, markersize=0.5)
    plt.show()

    print(A)
    B = dia_builder_Matrix(f,Ufx,Vfy,nx,ny)
    fig,ax = plt.subplots()
    ax.spy(B, markersize=0.5)
    plt.show()
    print(B.todense())
    print(np.linalg.norm(A-B))
    print(A-B)
    #plt.pauser()
    
    ## important for fast solves includes csr storage. 
    print("converting matrix to sparse row")
    t1_start = process_time()  
    As = sparse.csc_matrix(A)
    t1_stop = process_time()
    print("done in ",t1_stop - t1_start)
    
    for n in range(0,nt):

        #Animate3d(xxc, yyc, Phi, xmin, xmax, ymin, ymax)
        #Plotv(Phi)
        #f,q = scipy.sparse.linalg.lgmres(A,f,atol = 1e-10,tol = 1e-10) ## some sort of solver. 
        t1_start = process_time()  
        f,q = scipy.sparse.linalg.lgmres(As,f,tol = 1e-10)
        t1_stop = process_time()
        print("time_for_solve",t1_stop - t1_start,q)
        
        if n%10==0:
            plt.clf()
            #Animate3d(xxc, yyc,f.reshape(ny,nx), xmin, xmax, ymin, ymax)
            plt.contourf( f.reshape(ny,nx) , [-0.01,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.01])
            plt.draw()
            plt.pause(0.001)
    plt.show()
    
def initalCondition(xxc,yyc):
    return 0 + 1*(xxc>0.25)*(xxc<0.55)*(yyc>0.25)*(yyc<0.55)
    
def InitialVelocity(a,b):
    "input a,b, the (x,y) coordinate for a face, or quadrature points, in which you want velocity."
    #nn = 1
    #U = 2+a
    #V = 0+1*b#-1+ 0*a
    #U = np.sin(nn*np.pi*a)*np.cos(nn*np.pi*b)
    #V = -np.cos(nn*np.pi*a)*np.sin(nn*np.pi*b)
    U = -(b-0.5)
    V = (a-0.5)
    return U,V

def pos(c):
    return 0.5*(c+np.abs(c))

def neg(c):
    return 0.5*(c-np.abs(c))


def DenseBuilder(f,Ufx,Vfy,nx,ny):
    print("creating dense numpy array")
    t1_Build = process_time()  
    ### defining the matrices and vectors. I need to learn how to do this in sparse format.
    N = len(f)
    A = np.diag(1+ pos(Ufx) - Vecxroll(neg(Ufx),nx,ny,1) + pos(Vfy) - Vecyroll(neg(Vfy),nx,ny,1) )
    A += Matxroll(np.diag(neg(Ufx)),nx,ny,-1) #  upper
    A += - Matxroll(np.diag(Vecxroll(pos(Ufx),nx,ny,1)),nx,ny,1) ## lower
    A += Matyroll(np.diag(neg(Vfy)),nx,ny,-1) # upper nx
    A += - Matyroll(np.diag(Vecyroll(pos(Vfy),nx,ny,1)),nx,ny,1) ## lower nx
    t1_Builded = process_time()
    print(" 2nd done in ",t1_Build - t1_Builded)    
    return A
    
def dia_builder_Matrix(f,Ufx,Vfy,nx,ny):
    """i am using vec xroll where maybe inappropriate"""

    print("creating dense numpy array")
    t1_Build = process_time()  
    
    N = len(f)
    S = dia_matrix( (N,N),dtype = float)
    main = 1+ pos(Ufx) - Vecxroll(neg(Ufx),nx,ny,1) + pos(Vfy) - Vecyroll(neg(Vfy),nx,ny,1) 
    lower = -Vecxroll(pos(Ufx),nx,ny,1)
    upper = neg(Ufx)
    lowernx = -Vecyroll(pos(Vfy),nx,ny,1) 
    uppernx = neg(Vfy)
    
    upper = np.roll(upper,-1)
    #lower = np.roll(lower,-1)
    #lowernx = np.roll(lowernx,-nx)
    
    S.setdiag(main, k=0)
    S.setdiag(upper, k=1)
    S.setdiag(lower, k=-1)
    S.setdiag(uppernx, k=nx)
    S.setdiag(lowernx, k=-nx)
    
    S.setdiag(lowernx[N-nx:N], k=+(N-nx))
    S.setdiag(uppernx[N-nx:N], k=-(N-nx))
    S.setdiag(upper[0], k=-(N-1))
    S.setdiag(lower[-1], k=(N-1))
    

    t1_Builded = process_time()
    print(" sparse done in ",t1_Build - t1_Builded)
    return S    
    

def Vecxroll(v,nx,ny,Roll):
    t1_start = process_time()  
    for j in range(0,ny):
        v[j*nx:(j+1)*nx]  = np.roll(v[j*nx:(j+1)*nx],Roll)
    t1_stop = process_time()
    print("time_for_vecxroll",t1_stop - t1_start)
    return v
    
def Vecyroll(v,nx,ny,Roll):
    t1_start = process_time()  
    v  = np.roll(v,Roll*nx)
    t1_stop = process_time()
    print("time_for_vecyroll",t1_stop - t1_start)
    return v
    
def Matxroll(A,nx,ny,Roll):
    t1_start = process_time()  
    for j in range(0,ny):
        A[:,j*nx:(j+1)*nx]= np.roll( A[:,j*nx:(j+1)*nx],Roll,axis = 0 )
    t1_stop = process_time()
    print("time_for_Matxroll",t1_stop - t1_start)
    return A
    
def Matyroll(A,nx,ny,Roll):
    t1_start = process_time()  
    A = np.roll(A,Roll*nx,axis = 0)
    t1_stop = process_time()
    print("time_for_Matyroll",t1_stop - t1_start)
    return A
    
def SCIPYxROLL(B,Roll): ### scipy doesnt have a roll
    return scipy.sparse.hstack([B[:,Roll:],B[:,:Roll]])

def SciMatxroll(A,nx,ny,Roll): ### scipy doesnt have a roll
    Roll = Roll
    t1_start = process_time()  
    for j in range(0,ny):
        A[:,j*nx:(j+1)*nx]=  SCIPYxROLL(A[:,j*nx:(j+1)*nx],Roll)
    t1_stop = process_time()
    print("time_for_SciMatxroll",t1_stop - t1_start)
    return A
    
def SciMatyroll(A,nx,ny,Roll): ### scipy doesnt have a roll
    t1_start = process_time()  
    A = SCIPYxROLL(A,Roll*nx)
    t1_stop = process_time()
    print("time_for_SciMatyroll",t1_stop - t1_start)
    return A

if __name__ == "__main__":
    __main__()

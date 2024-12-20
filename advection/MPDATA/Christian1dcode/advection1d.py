#! /usr/bin/env python
# Copyright (C) 2019 ECMWF
# adapted from NM training course lectures

"""
run.py

Python program to simulate 1D linear advection

Note:
  This program could be made more efficient in different programming
  languages, or different implementations. It is intended for 
  educational use only, with as much clarity as possible.
"""

# Imports
from numpy import pi, sin, exp, sqrt, min, max, abs, linspace, empty
from time  import clock
from argparse import ArgumentParser

# ------------------------------------------------------------------------------------
#
# Default Setup
#
N         = 99
xmin      = 0.
xmax      = 10.
tend      = 100.
dt        = 0.05
a         = 1.
init      = 'gaussian' # Possible values:  sin, gaussian, heavy
scheme    = 'default'  # Possible values:  upstream,LW,MPDATA
output_file = 'out'
final_plot   = True
write_output = True

# ------------------------------------------------------------------------------------

# Allow override from command-line arguments
parser = ArgumentParser()
parser.add_argument('--N',     type=int,   help='Number of points',       default=N    )
parser.add_argument('--xmin',  type=float, help='Minimum X value',        default=xmin )
parser.add_argument('--xmax',  type=float, help='Maximum X value',        default=xmax )
parser.add_argument('--tend',  type=float, help='End time of simulation', default=tend )
parser.add_argument('--dt',    type=float, help='Time step',              default=dt   )
parser.add_argument('--a',     type=float, help='Advection velocity',     default=a    )
parser.add_argument('--init',  type=str,   help='Initial condition. Possible values: gaussian, sin, heavy',      default=init )
parser.add_argument('--scheme',type=str,   help='Advection scheme. Possible values: upstream, LW, MPDATA',       default=scheme )
parser.add_argument('--out',   type=str,   help='Output file',            default=output_file )
parser.add_argument('--verbose',           help='Print progress',         action='store_true' )
args = parser.parse_args()

N         = args.N
xmin      = args.xmin
xmax      = args.xmax
tend      = args.tend
dt        = args.dt
a         = args.a
init      = args.init
scheme    = args.scheme
output_file = args.out

# ------------------------------------------------------------------------------------

def save( Q, t, n ):
    """
    Save variable to file
    """
    if( write_output ):
        with open( output_file, 'a' ) as out:
            print >>out, "# comment line"
            print >>out, N, t
            for p in range(N):
                print >>out, X[p], Q[p]

# ------------------------------------------------------------------------------------
   
def plot( Q, Q0, title ):
    """
    Plot solution using matplotlib
    """
    try: # See if we can import matplotlib
        import matplotlib.pyplot as plt
    except:
        print "Plotting with matplotlib not supported"
        return

    #plt.plot( X,   Q,    '-',   hold=True )
    #plt.plot( X,   Q0,   'k-',  hold=True )
    plt.plot( X,   Q,    '-' )
    plt.plot( X,   Q0,   'k-')
    plt.title(title)
    plt.grid( True )
    plt.show()

# ------------------------------------------------------------------------------------

def flux( q, u ):
    """
    Analytical flux function
    """
    return u*q

# ------------------------------------------------------------------------------------

def upstream_flux( q_m, q_p , u):
    """
    Upwind flux function
    """
    return 0.5*( flux(q_m,u) + flux(q_p,u) ) - 0.5*abs(u)*(q_p-q_m)

# ------------------------------------------------------------------------------------

def error_norm(Q,Qexact):
    """
    Compute error norm L2
    """
    err = empty(N)
    errnorm = 0
    for j in range(N):
        errnorm = errnorm+(Q[j]-Qexact[j])**2
    errnorm = sqrt(errnorm)/N
    return errnorm

# ------------------------------------------------------------------------------------

def Upstream(Q,t,dt):
    """
    Forward in time, Forward in space advection scheme (1st order)
    """
    Qnp1 = empty(N)
    if( a>0 ):
        for j in range(1,N):
           Qnp1[j] = Q[j] - dt*( a*Q[j] - a*Q[j-1] )/dx
        Qnp1[0] = Q[0] - dt*( a*Q[0]-a*Q[N-1] )/dx
    else:
        for j in range(0,N-1):
            Qnp1[j] = Q[j] - dt*( a*Q[j+1] - a*Q[j  ] )/dx
        Qnp1[N-1] = Q[N-1] - dt*( a*Q[0  ] - a*Q[N-1] )/dx
    return Qnp1

# ------------------------------------------------------------------------------------

def LaxWendroff(Q,t,dt):
    """
    Lax-Wendroff advection scheme
    """
    Qnp1 = empty(N)
    # Add code here
    for j in range(1,N-1):
        Qnp1[j] = Q[j] - dt*( a*Q[j+1] - a*Q[j-1] )/(2*dx) + dt*dt*a*a*(Q[j+1] - 2*Q[j] + Q[j-1])/(2*dx*dx)
    Qnp1[0]   = Q[0]   - dt*( a*Q[1]-a*Q[N-1] )/(2*dx) + dt*dt*a*a*(Q[1] - 2*Q[0] + Q[N-1])/(2*dx*dx)
    Qnp1[N-1] = Q[N-1] - dt*( a*Q[0]-a*Q[N-2] )/(2*dx) + dt*dt*a*a*(Q[0] - 2*Q[N-1] + Q[N-2])/(2*dx*dx)
    return Qnp1

# ------------------------------------------------------------------------------------

def MPDATA(Q,t,dt):
    """
    MPDATA advection scheme
    """
    Qnp1 = empty(N)
    fx = empty(N+1)
    vx = empty(N+1)
 
    # first-order upwind pass
    for j in range(1,N):
        fx[j] = upstream_flux(Q[j-1],Q[j],a)
    fx[0] = upstream_flux(Q[N-1],Q[0],a)
    fx[N] = fx[0]
    for j in range(0,N):
        Q[j] = Q[j] - dt*(fx[j+1]-fx[j])/dx
   
    # anti-diffusive velocity 
    for j in range(1,N):
        vx[j] = anti_diffusive_velocity(Q[j-1],Q[j],a,dt)
    vx[0] = anti_diffusive_velocity(Q[N-1],Q[0],a,dt) 
    vx[N] = vx[0]
 
    # corrective pass using anti-diffusive velocity
    for j in range(1,N):
        fx[j] = upstream_flux(Q[j-1],Q[j],vx[j])
    fx[0] = upstream_flux(Q[N-1],Q[0],vx[0])
    fx[N] = fx[0]
    for j in range(0,N):
        Q[j] = Q[j] - dt*(fx[j+1]-fx[j])/dx

    return Q
    
def anti_diffusive_velocity(q1,q2,u,dt):
    return (abs(u)-u*u*dt/dx)*(q2-q1)/(q2+q1+1.e-10)

# ------------------------------------------------------------------------------------

"""
Main program
"""

X = linspace( xmin, xmax, num=N, endpoint=False )
dx = (xmax-xmin)/N
cfl = dt/dx*abs(a)

if( write_output ):
  f = open( output_file, 'w' )
  f.close()

# Initial condition
print 'Set initial condition'
if (init == 'sin'):         # Sin wave
    Q = sin(2.*pi * X/(xmax-xmin))
elif (init == 'gaussian'):  # Gaussian wave
    mu  = 0.5*(xmin+xmax)
    sig = 0.1*(xmax-xmin)
    Q = exp( -(X-mu)**2 / (2.*sig**2) )
elif (init == 'heavy'):  # Heavy
    mu  = 0.5*(xmin+xmax)
    sig = 0.1*(xmax-xmin)
    Q = 0*X
    for n in range(N):
        if ( abs(mu-X[n])<sig ):
            Q[n] = 1.
else:
    Q = 0.*X

Q0 = Q.copy()

# Time stepping
t = 0.
n = 0
print 'Start time stepping'
save(Q,t,n)
start = clock()

while( t < tend ):

    # To make simulation end exactly at tend
    DT=min([dt,tend-t])

    # Propagate one time step
    if( scheme=='upstream' ):
        Q = Upstream(Q,t,DT)
    elif( scheme == 'LW' ):
        Q = LaxWendroff(Q,t,DT)
    elif( scheme == 'MPDATA'):
        Q = MPDATA(Q,t,DT)
    else:
        scheme = 'default'
        Q = Upstream(Q,t,DT)

    t += DT
    n += 1
    if( args.verbose ):
        print 'iter ['+str(n)+']  time ['+str(t)+']'
    save( Q, t, n )

elapsed = clock() - start
print 'Time stepping took',elapsed,'seconds'
if( write_output ):
    print 'Turn off output for accurate timing'
errnorm = error_norm(Q,Q0)

# Summary of used simulation parameters
print '----------------------------------'
print '  Final time : ',t
print '  Time step  : ',dt
print '  CFL        : ',cfl
print '  Nb points  : ',N
print '  Simulation steps : ',n
print '  Simulation time  : ',elapsed,'s'
print '  ERROR NORM : ', errnorm
print '----------------------------------'

# Plotting
if( final_plot ):
    plot( Q, Q0, title='scheme='+scheme+';  a='+str(a)+';  t='+str(t)+' s;  dt='+str(dt)+';  N='+str(N)+';  cfl='+str(cfl))

print 'Exit program'

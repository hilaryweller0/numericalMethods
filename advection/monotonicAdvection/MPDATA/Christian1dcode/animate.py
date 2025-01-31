#!/usr/bin/env python

try: # See if we can import matplotlib
    import matplotlib.pyplot as plt
except:
    print "Plotting with matplotlib not supported"

import sys
import time
import pylab
from numpy import empty

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--skip', type=int, help='skip iterations', default=1)
parser.add_argument('--fps', type=int, help='frames per second', default=20)
parser.add_argument('--begin', help='Always plot first frame', action='store_true')
parser.add_argument('--end', help='Only plot last frame', action='store_true')
parser.add_argument('--ymin', type=float, help='y-axis min value', default=-0.25)
parser.add_argument('--ymax', type=float, help='y-axis max value', default=1.25)
parser.add_argument("file", type=str, nargs='?', help="input file", default='out')
args = parser.parse_args()

fname = args.file
f     = open(fname,"r")

pylab.ion()

nskip = args.skip
sleeptime = 1./args.fps

ymin = args.ymin
ymax = args.ymax

def plot(x,y,t,clear=True):
    plt.ion()
    if( clear ):
       pylab.clf()
       pass
    plt.plot(x, y)
    plt.axis([x[0],x[-1],ymin,ymax])
    plt.grid(True)
    plt.title('t = '+str(t))


n=0
step=0
x0 = []
y0 = []
while(1):
    line = f.readline()
    line = f.readline()
    if( not line ):
        break

    line = line.split()

    N       = int(line[0])
    t       = float(line[1])

    xlist = empty(N)
    ylist = empty(N)
    for j in range(N):
        line = f.readline()
        xy = line.split()
        xlist[j] = float(xy[0])
        ylist[j] = float(xy[1])

    if( step==0 and args.begin ):
        x0 = xlist.copy()
        y0 = ylist.copy()

    if(n==0 and not args.end):
        clr=True
        if( args.begin ):
            plot(x0,y0,t)
            clr=False
        plot(xlist,ylist,t,clear=clr)
        plt.draw()
        plt.pause(sleeptime)
        #pylab.savefig('out'+str(step)+'.png')
    n += 1
    if( n==nskip ):
        n=0
    step += 1

if( args.end ):
    clr=True
    if( args.begin ):
        plot(x0,y0,t)
        clr=False
    plot(xlist,ylist,t,clear=clr)

pylab.ioff()
plt.show()






#Discontinuous_ic_issue.py
#Carter Johnson
#MAT228B Assignment 1

#Illustrate a problem with Crank-Nicolson method
#on a diffusion problem with discontinuous initial condition

#u_t = u_xx, 0<x<1
#u(0,t) =1,  u(1,t) = 0
#u(x,0) = 1 if x<0.5, 0 if x>= 0.5

from __future__ import division

import numpy as np
from numpy import exp

import matplotlib.pyplot as plt

from crank_nicolson import crank_nicolson_method
from bdf2 import bdf2_method

def illustrate_issue():
	#set up the vectors and parameters for Crank-Nicolson method and run
	#using diff coefficient, initial condition from problem 3
	#no forcing function, but include left BC as forcing function at x=0
	del_x = 0.02
	del_t = 0.1

	#make matrix of forcing function f=0 at all times and spaces
	Nx = int(1/del_x)-1
	Nt = int(1/del_t)-1
	f = np.zeros((Nt+1,Nx))
	#include RHS BC u(0,t)=1
	f[:,0] = 1/(del_x**2)*np.ones(Nt+1)
	
	#initial condition u(x,0)=1 if x<0.5, 0 if x>=0.5
	u = np.zeros(Nx)
	for i in range(int(Nx/2)):
		u[i] = 1

	#plot IC
	plt.plot(u)
	plt.show()
	plt.close()

	#diffusion coefficient
	D = 1
	u = crank_nicolson_method(del_x, del_t, u, f, D)
	
	#plot u(x,1)
	plt.plot(u)
	plt.show()
	plt.close()

def fix_issue():
	#set up the vectors and parameters for BDF-2 method and run
	#using diff coefficient, initial condition from problem 3
	#no forcing function, but include left BC as forcing function at x=0
	del_x = 0.02
	del_t = 0.1

	#make matrix of forcing function f=0 at all times and spaces
	Nx = int(1/del_x)-1
	Nt = int(1/del_t)-1
	f = np.zeros((Nt+1,Nx))
	#include RHS BC u(0,t)=1
	f[:,0] = 1/(del_x**2)*np.ones(Nt+1)
	
	#initial condition u(x,0)=1 if x<0.5, 0 if x>=0.5
	u = np.zeros(Nx)
	for i in range(int(Nx/2)):
		u[i] = 1

	#plot IC
	plt.plot(u)
	plt.show()
	plt.close()

	#diffusion coefficient
	D = 1
	u = bdf2_method(del_x, del_t, u, f, D)
	
	#plot u(x,1)
	plt.plot(u)
	plt.show()
	plt.close()

if __name__ == '__main__':
	illustrate_issue()
	fix_issue()
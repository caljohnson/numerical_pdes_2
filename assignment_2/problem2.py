#Problem2.py
#Carter Johnson

#MAT228B
#Assignment 2

#Runtime comparison of Forward Euler and Crank-Nicolson
#for 2-D diffusion eqn

#u_t = ∆u  on Ω = (0,1)x(0,1)
#u = 0 on ∂Ω
#u(x,y,0)=exp(-100((x-0.3)^2+(y-0.4)^2))

from __future__ import division

import numpy as np
from numpy import exp, sin, pi
from numpy.linalg import norm
import matplotlib.pyplot as plt

from tabulate import tabulate
from tqdm import tqdm

from crank_nicolson_2D import crank_nicolson_method

def run_Crank_Nicolson_problem2(h,del_t):
	#run Crank-Nicolson solver on the problem
	#set up the vectors and parameters for Crank-Nicolson method and run
	#using diffusion coefficient, initial condition, forcing function from problem 2

	#make vector of forcing function at all times 
	Nx = int(1/h)-1
	Ny = int(1/h)-1
	Nt = int(1/del_t)
	X = [i*h for i in range(1, Nx+1)]
	Y = [i*h for i in range(1, Ny+1)]
	t = [i*del_t for i in range(0, Nt+1)]
	
	#f = 0
	f = [0*t for t in t]
	
	#initial condition u(x,y,0)=exp(-100((x-0.3)^2+(y-0.4)^2))
	u = [[exp(-100*((x-0.3)**2+(y-0.4)**2)) for x in X] for y in Y]
	u = np.asarray(u)
	print(u)
	print(norm(u))
	#diffusion coefficient
	D = 1
	u = crank_nicolson_method(h, del_t, u, f, D)

	print(u)
	print(norm(u))
	return u	

if __name__ == '__main__':
	# errors_refinement_study()
	run_Crank_Nicolson_problem2(0.25,0.01)


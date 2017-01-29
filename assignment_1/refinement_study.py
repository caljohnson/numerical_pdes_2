#Refinement_study.py
#Carter Johnson
#MAT228B Assignment 1

#Refinement Study for
#1-D Diffusion equation solver using Crank-Nicolson routine
#3-pt 2nd order spatial discretization 
#with a trapezoidal rule for time
#for Dirichlet BC's

#for the problem
#u_t = 0.01 u_xx + 1 - exp(-t)
#0 < x < 1
#u(0,t) = u(1,t) = 0
#u(x,0) = 0

from __future__ import division

import numpy as np
from numpy import exp
from numpy.linalg import norm

from tabulate import tabulate
from tqdm import tqdm

from crank_nicolson import crank_nicolson_method

def refinement_study():
	#perform a refinement study to demonstrate Crank-Nicolson
	#is second-order accurate in space and time

	#SHOW 2nd order in space

	#max number of del_x values to examine
	refine_MAX = 15

	#set time step
	del_t = 0.01

	#loop through del_x values
	del_x = [2**(-1-i) for i in range(0,refine_MAX)]

	#set container for successive differences
	diffs = np.zeros(refine_MAX)

	#get u(x,1) through Crank-Nicolson:
	u_new = setup_and_run(del_x[0], del_t)

	#loop over finer del_x, take successive differences
	for i in tqdm(range(1,refine_MAX)):
		#store previous iterate
		u_old = u_new + 0

		#get next u(x,1) through Crank-Nicolson
		u_new = setup_and_run(del_x[i], del_t)

		#calculate successive difference
		diffs[i] = abs(norm(u_old,2)- norm(u_new,2))

	two_norm_table = [[del_x[i], diffs[i], diffs[i+1]/diffs[i]] for i in range(refine_MAX-1)]	
	print(tabulate(two_norm_table, headers=['\delta x', 'diffs', 'diff ratios'], tablefmt="latex"))

def setup_and_run(del_x, del_t):
	#set up the vectors and parameters for Crank-Nicolson method and run

	#make vector of forcing function at all times 
	Nx = int(1/del_x)-1
	Nt = int(1/del_t)-1
	x = [i*del_x for i in range(0, Nx+1)]
	t = [i*del_t for i in range(0, Nt+1)]
	
	#f = 1-exp(-t)
	neg_t = [-t for t in t] 
	f = 1-exp(neg_t)
	
	#initial condition u(x,0)=0
	u = np.zeros(Nx)

	#diffusion coefficient
	D = 0.01
	u = crank_nicolson_method(del_x, del_t, u, f, D)
	return u	

if __name__ == '__main__':
	refinement_study()
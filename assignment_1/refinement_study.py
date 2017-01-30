#Refinement_study.py
#Carter Johnson
#MAT228B Assignment 1

#Refinement Study 
#1-D Diffusion equation solver using Crank-Nicolson routine
#3-pt 2nd order spatial discretization 
#with a trapezoidal rule for time
#for Dirichlet BC's

#for Problem 2
#u_t = 0.01 u_xx + 1 - exp(-t)
#0 < x < 1
#u(0,t) = u(1,t) = 0
#u(x,0) = 0

from __future__ import division

import numpy as np
from numpy import exp
from numpy.linalg import norm
import matplotlib.pyplot as plt

from tabulate import tabulate
from tqdm import tqdm

from crank_nicolson import crank_nicolson_method
from bdf2 import bdf2_method

def refinement_study():
	#perform a refinement study to demonstrate Crank-Nicolson
	#is second-order accurate in space and time

	#SHOW 2nd order in space

	#max number of del_x,del_t values to examine
	refine_MAX = 10

	#loop through del_x values
	del_x = [2**(-1-i) for i in range(0,refine_MAX)]
	del_t = [2**(-1-i) for i in range(0, refine_MAX)]

	#set container for successive differences
	diffs = np.zeros(refine_MAX)

	#get u(x,1) through Crank-Nicolson:
	u_new = setup_and_run(del_x[0], del_t[0])
	# #plot u(x,1)
	# plt.plot(u_new)
	# plt.show()
	# plt.close()	

	#loop over finer del_x, take successive differences
	for i in tqdm(range(1,refine_MAX)):
		#store previous iterate
		u_old = u_new + 0

		#get next u(x,1) through Crank-Nicolson
		u_new = setup_and_run(del_x[i], del_t[i])
		
		#calculate successive difference between u(x,1) new and old	
		diffs[i] = interpolate_diffs(u_new, u_old, del_x[i])

		#plot u(x,1)
		# plt.plot(u_new)
		# plt.show()
		# plt.close()	

	print(u_new)

	two_norm_table = [[del_x[i], del_t[i], diffs[i], diffs[i]/diffs[i+1]] for i in range(refine_MAX-1)]	
	print(tabulate(two_norm_table, headers=['delta x', 'delta t', 'diffs', 'diff ratios'], tablefmt="latex"))

def interpolate_diffs(u_new, u_old, h):
	#interpolate both u_new and u_old on a finer grid
	#and take the 2-norm difference of the interpolations
	interpol_new = interpolate(u_new, h)
	interpol_old = interpolate(interpolate(u_old, 2*h),h)
	return norm(interpol_old-interpol_new)

def interpolate(u_c, h):
	#interpolate to a 2x finer mesh
	n = int(1/h)-1
	h2 = h/2
	n2 = int(1/h2)-1
	u_f = np.zeros(n2+1, dtype=float)

	#loop over coarse mesh
	for i in range(n):
		u_f[2*i] = u_c[i]
		u_f[2*i-1] += u_c[i]/2
		u_f[2*i+1] += u_c[i]/2

	return u_f

def restriction(u_f, h):
	#simple restriction operation
	h2 = 2*h
	n2 = int(1/h2)-1
	u_c = np.zeros(n2, dtype=float)

	#loop over coarse mesh
	for i in range(0,n2):
		u_c[i] = u_f[2*i+1]
		
	return u_c

def setup_and_run(del_x, del_t):
	#set up the vectors and parameters for Crank-Nicolson method and run
	#using diffusion coefficient, initial condition, forcing function from problem 2

	#make vector of forcing function at all times 
	Nx = int(1/del_x)-1
	Nt = int(1/del_t)-1
	# x = [i*del_x for i in range(0, Nx+1)]
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
	# u = [1, 2, 3, 4, 5, 6, 7]
	# u = restriction(u,1/8)
	# print(u)
	# raise
	refinement_study()
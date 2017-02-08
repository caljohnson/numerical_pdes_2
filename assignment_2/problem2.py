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
from time import clock

from crank_nicolson_2D import crank_nicolson_method
from forward_euler_diffusion_2d import forward_euler_method

def run_comparison():
	#run comparison between Crank-Nicolson and Forward euler on the problem
	#set up the vectors and parameters for the methods and run
	#using diffusion coefficient, initial condition, forcing function from problem 2

	#set vector of grid spacings
	h = [2**(-i) for i in range(1,8)]

	#set time steps for CN, forcing function for CN
	CN_del_t = 0.01
	Nt_CN = int(1/CN_del_t)
	f_CN = np.zeros(Nt_CN+1)

	#diffusion coefficient
	D = 1

	#Don't plot
	plotting=0

	#record run times
	CN_times = []
	FE_times = []
	
	for i in range(len(h)):

		FE_del_t = h[i]**2/4

		#make vector of forcing function for FE at all times, also get grid points for level h
		Nx = int(1/h[i])-1
		Ny = int(1/h[i])-1
		Nt_FE = int(1/FE_del_t)
		X = [j*h[i]for j in range(1, Nx+1)]
		Y = [j*h[i] for j in range(1, Ny+1)]
		
		#f =0
		f_FE = np.zeros(Nt_FE+1)
	
		#initial condition u(x,y,0)=exp(-100((x-0.3)^2+(y-0.4)^2))
		u = [[exp(-100*((x-0.3)**2+(y-0.4)**2)) for x in X] for y in Y]
		u = np.asarray(u)


		toc=clock()
		u = crank_nicolson_method(h[i], CN_del_t, u, f_CN, D, X, Y, plotting)
		tic=clock()
		CN_times.append(tic-toc)

		toc=clock()
		u = forward_euler_method(h[i], FE_del_t, u, f_FE, D, X, Y, plotting)
		tic=clock()
		FE_times.append(tic-toc)

	runtime_table = [[h[i], CN_times[i], FE_times[i]] for i in range(len(h))]
	print(tabulate(runtime_table, headers=["grid spacing", "Crank-Nicolson Run time", "Forward Euler run time"]))
	return u	


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
	print(norm(u))

	#Decide whether or not to plot, 1=plot, 0=dont plot
	plotting =1

	#diffusion coefficient
	D = 1

	#GO
	u = crank_nicolson_method(h, del_t, u, f, D, X, Y, plotting)

	print(norm(u))
	return u	

def run_Forward_Euler_problem2(h,del_t):
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
	print(norm(u))

	#Decide whether or not to plot, 1=plot, 0=dont plot
	plotting =1

	#diffusion coefficient
	D = 1

	#GO
	u = forward_euler_method(h, del_t, u, f, D, X, Y, plotting)

	print(norm(u))
	return u	


if __name__ == '__main__':
	run_comparison()
	# h = 2**(-3)
	# del_t = h**2/4
	# run_Crank_Nicolson_problem2(h, 0.01)
	# run_Forward_Euler_problem2(h, del_t)


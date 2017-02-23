#Peaceman_Rachford.py
#Carter Johnson
#Mat228B Assignment 3

#Peaceman-Rachford ADI scheme on a cell-centered grid
#for solving 2-d homogeneous diffusion eqn with Neumann BCs

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import exp
from numpy.linalg import norm
from tabulate import tabulate
from tqdm import tqdm
from time import clock
import scipy.sparse as sparse
import scipy.sparse.linalg

def sparse_matrices(h):
	#set sparse matrix L, the discrete 1-D Laplacian
	#for 3-pt centered flux 2nd order approximation
	#includes Neumann BCs

	#Set number of grid points
	N = int(1/h - 1)

	#set off-diagonal Laplacian components
	off_diag = 1*np.ones(N)
	#set diagonal Laplacian components
	diag = (-2)*np.ones(N)
	diag[0] = -1
	diag[-1] = -1

	# Generate the diagonal and off-diagonal matrices
	A = np.vstack((off_diag, diag, off_diag))/(h**2)
	L = scipy.sparse.dia_matrix((A,[-1,0,1]),shape=(N,N))
	I = scipy.sparse.identity(N)

	return L, I

def peaceman_rachford_step(u,h,delT,b,L,I):
	#one full time step of the ADI scheme
	N = int(1/h -1)

	#Diffuse in x direction
	#iterate over columns of u^n to get columns of u^*
	u_star = np.zeros((N,N))
	for i in range(N):
		#get column of u^n
		u_col = u[:,i]
		#(I + b*delT/2 L_y)u^n
		A = (I + (b*delT/2) * L)
		RHS_terms = A.dot(u_col)

		#make LHS matrix, put in CSC form for solver
		LHS_matrix = scipy.sparse.csc_matrix(I-(b*delT/2)*L)

		#solve (I - b*delT/2 L_x)u^* = (I + b*delT/2 L_y)u^n
		u_star[:,i] = scipy.sparse.linalg.spsolve(LHS_matrix, RHS_terms)

	#Diffuse in y direction
	#iterate over rows of u^* to get rows of u^n+1
	u_next = np.zeros((N,N))
	for i in range(N):
		#get row of u^*
		u_row = u_star[i,:]
		#(I + b*delT/2 L_x)u^*
		A = (I + (b*delT/2) * L)
		RHS_terms = A.dot(np.transpose(u_row))

		#make LHS matrix, put in CSC form for solver
		LHS_matrix = scipy.sparse.csc_matrix(I-(b*delT/2)*L)

		#solve (I - b*delT/2 L_y)u^n+1 = (I + b*delT/2 L_x)u^*
		u_next[i,:] = np.transpose(scipy.sparse.linalg.spsolve(LHS_matrix, RHS_terms))

	return u_next	

def peaceman_rachford_method(h,delT,b,u_old, plotting):
	N = int(1/h - 1)
	Nt = int(1/delT)

	#get operators
	[L, I] = sparse_matrices(h)

	energy = np.sum(u_old)
	
	if plotting==1:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		# `plot_surface` expects `x` and `y` data to be 2D
		grid_X = [h*(i-0.5) for i in range(1,N+1)]
		grid_Y = [h*(j-0.5) for j in range(1, N+1)]
		X, Y = np.meshgrid(grid_X, grid_Y)  
		#keep z limits fixed
		ax.set_zlim(0, 1)
		plt.ion()
		#plot first frame, u(x,y,0)
		frame = ax.plot_surface(X, Y, u_old)
		plt.pause(0.05)


	for t in range(Nt):
		#solve for next u
		u_new = peaceman_rachford_step(u_old, h, delT, b, L, I)
		
		if plotting==1:
			#plot current u
			ax.collections.remove(frame)
			frame = ax.plot_surface(X, Y, u_new)
			plt.pause(0.05)

		u_old = u_new + 0


	energy2 = np.sum(u_new)
	print(energy-energy2)
	return u_new

def refinement_study():
	#refinement study for 2d diffusion using peaceman-rachford ADI scheme
	#for homogenous diffusion w/ Neumann bcs

	#set vector of grid spacings/time steps
	h = [2**(-i) for i in range(1,10)]

	#diffusion coefficient
	b = 0.1

	#Don't plot
	plotting=0

	#record successive differences + ratios, run times and runtime ratios
	diffs = np.zeros(len(h))
	diff_ratios = np.zeros(len(h))
	times = np.zeros(len(h))
	time_ratios = np.zeros(len(h))

	
	for i in tqdm(range(len(h))):
		#get time step
		delT = h[i]

		#get grid points for level h
		N = int(1/h[i]-1)
		Nt = int(1/delT)
		X = [h[i]*(j-0.5) for j in range(1,N+1)]
		Y = [h[i]*(j-0.5) for j in range(1, N+1)]

		#initial condition u(x,y,0)=exp(-100((x-0.3)^2+(y-0.4)^2))
		u = [[exp(-100*((x-0.3)**2+(y-0.4)**2)) for x in X] for y in Y]
		u = np.asarray(u)


		toc=clock()
		u_new = peaceman_rachford_method(h[i], delT, b, u, plotting)
		tic=clock()
		if i>0:
			diffs[i]=(h[i-1]**2)*norm(restriction(u_new, h[i]) - u_old,ord=1)
			time_ratios[i] = (tic-toc)/times[i-1]
		if i>1:
			diff_ratios[i]=diffs[i-1]/diffs[i]
		u_old = u_new+0	
		times[i]=tic-toc


	table = [[h[i], times[i], time_ratios[i], diffs[i], diff_ratios[i]] for i in range(len(h))]
	print(tabulate(table, headers=["grid spacings/time steps", "Runtimes", "Runtime Ratios", "Successive Differences", "Difference Ratios"], tablefmt="latex"))

def restriction(u, h):
	u_f = u +0
	h2 = 2*h
	n2 = int(1/h2)-1
	u_c = np.zeros((n2, n2), dtype=float)

	#loop over coarse mesh
	for i in range(0,n2):
		for j in range(0,n2):
			u_c[i][j] = u_f[2*i+1][2*j+1]
	return u_c

def test():
	h = 2**(-8)
	delT = 2**(-4)
	N = int(1/h - 1)
	Nt = int(1/delT)
	plotting = 1

	[L, I] = sparse_matrices(h)

	grid_X = [h*(i-0.5) for i in range(1,N+1)]
	grid_Y = [h*(j-0.5) for j in range(1, N+1)]

	u0 = np.asarray([[exp(-10*((x-0.3)**2 + (y-0.4)**2)) for x in grid_X] for y in grid_Y])
	energy = np.sum(u0)

	u = peaceman_rachford_method(h,delT, 0.1, u0,1)

if __name__ == '__main__':
	# test()
	refinement_study()
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
	# iterate over columns of u^n to get columns of u^*
	A = (I + (b*delT/2) * L)
	RHS_terms = A.dot(u)
	LHS_matrix = scipy.sparse.csc_matrix(I-(b*delT/2)*L)
	u_star = scipy.sparse.linalg.spsolve(LHS_matrix, RHS_terms)	

	#Diffuse in y direction
	#iterate over rows of u^* to get rows of u^n+1
	A = (I + (b*delT/2) * L)
	RHS_terms = A.dot(np.transpose(u))
	LHS_matrix = scipy.sparse.csc_matrix(I-(b*delT/2)*L)
	u_next = np.transpose(scipy.sparse.linalg.spsolve(LHS_matrix, RHS_terms))

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
		# print(energy-energy2)


	# energy2 = np.sum(u_new)
	# print(energy-energy2)
	return u_new



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
	test()
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
from multiprocessing import Pool
from itertools import repeat
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
	N = int(1/h)-1
	#Diffuse in x direction
	# iterate over columns of u^n to get columns of u^*
	A = (I + (b*delT/2) * L)
	RHS_terms = A.dot(u)
	LHS_matrix = scipy.sparse.csc_matrix(I-(b*delT/2)*L)
	u_star = scipy.sparse.linalg.spsolve(LHS_matrix, RHS_terms)	

	# u_star2 = np.zeros((N,N))
	# for i in range(N):
	# 	#get column of u^n
	# 	u_col = u[:,i]
	# 	#(I + b*delT/2 L_y)u^n
	# 	A = (I + (b*delT/2) * L)
	# 	RHS_terms = A.dot(u_col)
	# 	#make LHS matrix, put in CSC form for solver
	# 	LHS_matrix = scipy.sparse.csc_matrix(I-(b*delT/2)*L)
	# 	#solve (I - b*delT/2 L_x)u^* = (I + b*delT/2 L_y)u^n
	# 	u_star2[:,i] = scipy.sparse.linalg.spsolve(LHS_matrix, RHS_terms)
	# print(norm(u_star2-u_star))

	#Diffuse in y direction
	#iterate over rows of u^* to get rows of u^n+1
	A = (I + (b*delT/2) * L)
	RHS_terms = A.dot(np.transpose(u_star))
	LHS_matrix = scipy.sparse.csc_matrix(I-(b*delT/2)*L)
	u_next = np.transpose(scipy.sparse.linalg.spsolve(LHS_matrix, RHS_terms))

	# u_next2 = np.zeros((N,N))
	# for i in range(N):
	# 	#get row of u^*
	# 	u_row = u_star[i,:]
	# 	#(I + b*delT/2 L_x)u^*
	# 	A = (I + (b*delT/2) * L)
	# 	RHS_terms = A.dot(u_row)

	# 	#make LHS matrix, put in CSC form for solver
	# 	LHS_matrix = scipy.sparse.csc_matrix(I-(b*delT/2)*L)

	# 	#solve (I - b*delT/2 L_y)u^n+1 = (I + b*delT/2 L_x)u^*
	# 	u_next2[i,:] = scipy.sparse.linalg.spsolve(LHS_matrix, RHS_terms)

	# print(norm(u_next2-u_next))
	# raise

	return u_next

def peaceman_rachford_step_pooling(u,h,delT,b,L,I):
	#one full time step of the ADI scheme
	N = int(1/h -1)

	# #Diffuse in x direction
	# # iterate over columns of u^n to get columns of u^*
	# A = (I + (b*delT/2) * L)
	# RHS_terms = A.dot(u)
	# LHS_matrix = scipy.sparse.csc_matrix(I-(b*delT/2)*L)
	# u_star = scipy.sparse.linalg.spsolve(LHS_matrix, RHS_terms)	

	# #Diffuse in y direction
	# #iterate over rows of u^* to get rows of u^n+1
	# A = (I + (b*delT/2) * L)
	# RHS_terms = A.dot(np.transpose(u))
	# LHS_matrix = scipy.sparse.csc_matrix(I-(b*delT/2)*L)
	# u_next = scipy.sparse.linalg.spsolve(LHS_matrix, RHS_terms)

	pool = Pool()
	#Diffuse in x direction
	#iterate over columns of u^n to get columns of u^*
	u_star = np.zeros((N,N))
	args = [u[:,i] for i in range(N)]
	u_star = pool.starmap(col_solves,zip(args, repeat(delT), repeat(b), repeat(L),repeat(I)))
	u_star = np.reshape(u_star, (N,N))
	#Diffuse in y direction
	args = [u_star[i,:] for i in range(N)]
	u_next = pool.starmap(row_solves, zip(args,repeat(delT), repeat(b), repeat(L),repeat(I)))
	u_next = np.reshape(u_next, (N,N))
	# for i in range(N):
	# 	#get column of u^n
	# 	u_col = u[:,i]
	# 	#(I + b*delT/2 L_y)u^n
	# 	A = (I + (b*delT/2) * L)
	# 	RHS_terms = A.dot(u_col)

	# 	#make LHS matrix, put in CSC form for solver
	# 	LHS_matrix = scipy.sparse.csc_matrix(I-(b*delT/2)*L)

	# 	#solve (I - b*delT/2 L_x)u^* = (I + b*delT/2 L_y)u^n
	# 	u_star[:,i] = scipy.sparse.linalg.spsolve(LHS_matrix, RHS_terms)

	# #Diffuse in y direction
	# #iterate over rows of u^* to get rows of u^n+1
	# u_next = np.zeros((N,N))
	# for i in range(N):
	# 	#get row of u^*
	# 	u_row = u_star[i,:]
	# 	#(I + b*delT/2 L_x)u^*
	# 	A = (I + (b*delT/2) * L)
	# 	RHS_terms = A.dot(np.transpose(u_row))

	# 	#make LHS matrix, put in CSC form for solver
	# 	LHS_matrix = scipy.sparse.csc_matrix(I-(b*delT/2)*L)

	# 	#solve (I - b*delT/2 L_y)u^n+1 = (I + b*delT/2 L_x)u^*
	# 	u_next[i,:] = np.transpose(scipy.sparse.linalg.spsolve(LHS_matrix, RHS_terms))

	pool.close()
	return u_next	

def col_solves(u_col, delT,b,L,I):
	#(I + b*delT/2 L_y)u^n
	A = (I + (b*delT/2) * L)
	RHS_terms = A.dot(u_col)

	#make LHS matrix, put in CSC form for solver
	LHS_matrix = scipy.sparse.csc_matrix(I-(b*delT/2)*L)

	#solve (I - b*delT/2 L_x)u^* = (I + b*delT/2 L_y)u^n
	return scipy.sparse.linalg.spsolve(LHS_matrix, RHS_terms)

def row_solves(u_row, delT, b, L, I):
	#(I + b*delT/2 L_x)u^*
	A = (I + (b*delT/2) * L)
	RHS_terms = A.dot(np.transpose(u_row))
	#make LHS matrix, put in CSC form for solver
	LHS_matrix = scipy.sparse.csc_matrix(I-(b*delT/2)*L)
	#solve (I - b*delT/2 L_y)u^n+1 = (I + b*delT/2 L_x)u^*
	return np.transpose(scipy.sparse.linalg.spsolve(LHS_matrix, RHS_terms))


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


	energy2 = np.sum(u_new)
	print(energy-energy2)
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

def test2():
	A = np.array([[1,2],[0,1]])
	u = np.array([[1,2],[3,4]])
	N=2
	#Diffuse in x direction
	# iterate over columns of u^n to get columns of u^*
	u_star=A.dot(u)

	u_star2 = np.zeros((N,N))
	for i in range(N):
		#get column of u^n
		u_col = u[:,i]
		u_star2[:,i] = A.dot(u_col)

	print(norm(u_star2-u_star))
	#Diffuse in y direction
	#iterate over rows of u^* to get rows of u^n+1
	u_next = np.transpose(A.dot(np.transpose(u_star)))

	u_next2 = np.zeros((N,N))
	for i in range(N):
		#get row of u^*
		u_row = u_star[i,:]
		u_next2[i,:] = np.transpose(A.dot(np.transpose(u_row)))

	print(norm(u_next2-u_next))
	raise


if __name__ == '__main__':
	test()
	# test2()
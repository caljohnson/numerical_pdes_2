#BDF2.py
#Cater Johnson
#MAT228B Assignment 1

#1-D Diffusion equation solver using BDF-2 routine
#3-pt 2nd order spatial discretization 
#with a 2nd order 3-pt approximation for time derivative
#for Dirichlet BC's

from __future__ import division

import numpy as np

import scipy.sparse as sparse
import scipy.sparse.linalg

def sparse_matrices(del_x):
	#set sparse matrix L, the discrete Laplacian
	#for 3-pt 2nd order approximation

	#Set number of grid points
	N = 1/del_x - 1

	#set off-diagonal Laplacian components
	offdiag = (1/(del_x**2))*np.ones(N)
	#set diagonal Laplacian components
	diag = np.ones(N)*(-2/(del_x**2))

	#put diagonals together into sparse matrix format
	data = np.vstack((offdiag, diag,offdiag))
	L = sparse.dia_matrix((data, [-1, 0,1]), shape = (N,N))

	#create identity matrix of same size as L
	I = sparse.identity(N)

	return L, I

def bdf2_time_step(del_t, u_new, u_old, L, f, I):
	#one time step of BDF-2 solver

	#use past two iterates on RHS
	#4u^n -u^n-1 + 2del_t f^n+1
	rhs_terms = 4*u_new  - u_old + 2*del_t*f

	#make LHS matrix, put in CSC form for solver
	LHS_matrix = scipy.sparse.csc_matrix(3*I-2*del_t*L)

	#solve (3I-2del_t L)u^n+1 = 4u^n -u^n-1 + 2del_t f^n+1
	u_next = scipy.sparse.linalg.spsolve(LHS_matrix, rhs_terms)

	return u_next

def back_euler_step(del_t, u, L, f, I):
	#one time step of Backward-Euler solver

	#use last iterate on RHS
	#4u^n -u^n-1 + 2del_t f^n+1
	rhs_terms = u + del_t*f

	#make LHS matrix, put in CSC form for solver
	LHS_matrix = scipy.sparse.csc_matrix(I-del_t*L)

	#solve (3I-2del_t L)u^n+1 = 4u^n -u^n-1 + 2del_t f^n+1
	u_next = scipy.sparse.linalg.spsolve(LHS_matrix, rhs_terms)

	return u_next

def bdf2_method(del_x, del_t, u, f, D):

	#create sparse matrices for crank-nicolson method
	[L, I] = sparse_matrices(del_x)
	#add diffusion coefficient to Laplacian
	L = D*L

	#calculate number of time points between 0 and 1
	Nt = int(1/del_t)-1

	#solve for first step using Backward-Euler
	u_old = u + 0
	u_new = back_euler_step(del_t, u_old, D*L, f[1], I)
	
	#step through time looking backwards
	for t in range(1,Nt):

		#solve for next u using previous two iterates u_new and u_old
		u = bdf2_time_step(del_t, u_new, u_old, D*L, f[t+1], I)
		
		#store previous iterates
		u_old = u_new+0
		u_new = u+0

	return u

if __name__ == '__main__':
	print('does nothing alone, import as a method')


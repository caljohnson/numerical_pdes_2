#Crank-Nicolson.py
#Cater Johnson
#MAT228B Assignment 1

#1-D Diffusion equation solver using Crank-Nicolson routine
#3-pt 2nd order spatial discretization 
#with a trapezoidal rule for time
#for Dirichlet BC's
#for time t=0 to t=1

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

def crank_nicolson_time_step(del_t, u, L, f, I):
	#one time step of crank-nicolson solver

	#(I + del_t/2 L)u^n
	A = (I + (del_t/2) * L)
	RHS_terms = A.dot(u) + del_t*f

	#make LHS matrix, put in CSC form for solver
	LHS_matrix = scipy.sparse.csc_matrix(I-(del_t/2)*L)

	#solve (I-del_t/2 L)u^n+1 = (I + del_t/2 L)u^n + del_t f^n+1/2
	u_next = scipy.sparse.linalg.spsolve(LHS_matrix, RHS_terms)

	return u_next

def crank_nicolson_method(del_x, del_t, u, f, D):

	#create sparse matrices for crank-nicolson method
	[L, I] = sparse_matrices(del_x)

	#calculate number of time points after 0 up to 1 (inclusive)
	Nt = int(1/del_t)

	for t in range(0,Nt):
		#take half point of f for solve
		f_half = (f[t]+f[t+1])/2
		#solve for next u
		u = crank_nicolson_time_step(del_t, u, D*L, f_half, I)
		
	return u

if __name__ == '__main__':
	print('does nothing alone, import as a method')
	# [A,I] = sparse_matrices(0.25)
	# print(A.toarray())
	# setup_and_run(1/4,1/100)
	# refinement_study()


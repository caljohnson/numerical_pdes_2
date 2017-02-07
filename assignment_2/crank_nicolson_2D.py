#Crank-Nicolson.py
#Cater Johnson
#MAT228B Assignment 1

#2-D Diffusion equation solver using Crank-Nicolson routine
#5-pt 2nd order spatial discretization 
#with a trapezoidal rule for time
#for Dirichlet BC's
#for time t=0 to t=1

from __future__ import division

import numpy as np

import scipy.sparse as sparse
import scipy.sparse.linalg

def sparse_matrices(del_x):
	#set sparse matrix L, the discrete 2-D Laplacian
	#for 5-pt 2nd order approximation

	#Set number of grid points
	N = 1/del_x - 1

	#set off-diagonal Laplacian components
	off_diag = 1*np.ones(N)
	#set diagonal Laplacian components
	diag = (-2)*np.ones(N)

	# Generate the diagonal and off-diagonal matrices
	A = np.vstack((off_diag, diag, off_diag))
	lap1D = scipy.sparse.dia_matrix((A,[-1,0,1]),shape=(N,N))
	speye = scipy.sparse.identity(N)
	#put diagonals together into sparse matrix format
	L = (1/(del_x**2))*(scipy.sparse.kron(lap1D,speye) + scipy.sparse.kron(speye,lap1D))

	return L, speye

def crank_nicolson_time_step(del_t, u, L, f, I):
	#one time step of crank-nicolson solver

	#(I + del_t/2 L)u^n
	A = (I + (del_t/2) * L)
	RHS_terms = A.dot(u.flatten(order='C')) + del_t*f

	#make LHS matrix, put in CSC form for solver
	LHS_matrix = scipy.sparse.csc_matrix(I-(del_t/2)*L)

	#solve (I-del_t/2 L)u^n+1 = (I + del_t/2 L)u^n + del_t f^n+1/2
	u_next = scipy.sparse.linalg.spsolve(LHS_matrix, RHS_terms)
	u_next = np.reshape(u_next, (n, n))

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
	# print('does nothing alone, import as a method')
	[A,I] = sparse_matrices(0.25)
	print(A.toarray())
	# setup_and_run(1/4,1/100)
	# refinement_study()


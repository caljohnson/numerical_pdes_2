#Crank-Nicolson.py
#Cater Johnson
#MAT228B Assignment 1

#1-D Diffusion equation solver using Crank-Nicolson routine
#3-pt 2nd order spatial discretization 
#with a trapezoidal rule for time
#for Dirichlet BC's

from __future__ import division

import numpy as np
from numpy import exp

import matplotlib.pyplot as plt
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
	temp_u = (I + (del_t/2) * L)*u

	#make LHS matrix, put in CSC form for solver
	LHS_matrix = scipy.sparse.csc_matrix(I-(del_t/2)*L)

	#solve (I-del_t/2 L)u^n+1 = (I + del_t/2 L)u^n + del_t f^n+1/2
	u_next = scipy.sparse.linalg.spsolve(LHS_matrix, temp_u + del_t*f)

	return u_next

def crank_nicolson_method(del_x, del_t, u, f, D):

	#create sparse matrices for crank-nicolson method
	[L, I] = sparse_matrices(del_x)
	#add diffusion coefficient to Laplacian
	L = D*L

	#calculate number of time points between 0 and 1
	Nt = int(1/del_t)-1

	for t in range(0,Nt):
		#take half point of f for solve
		f_half = (f[t]+f[t+1])/2
		#solve for next u
		u = crank_nicolson_time_step(del_t, u, D*L, f_half, I)
		plt.plot(u)

	return u

def setup():
	#set up the vectors and parameters for Crank-Nicolson method
	#Numerical scheme parameters
	del_x = 1/16
	del_t = 0.01

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
	print(u)

if __name__ == '__main__':
	setup()



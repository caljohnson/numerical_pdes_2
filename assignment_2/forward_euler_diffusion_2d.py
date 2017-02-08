#Forward_Euler_Diffusion_2D.py
#Cater Johnson
#MAT228B Assignment 2

#2-D Diffusion equation solver using Forward Euler
#using 5-pt 2nd order spatial discretization
#for Dirichlet BC's
#for time t=0 to t=1

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import scipy.sparse as sparse
import scipy.sparse.linalg

def sparse_matrices(h):
	#set sparse matrix L, the discrete 2-D Laplacian
	#for 5-pt 2nd order approximation

	#Set number of grid points
	N = 1/h - 1

	#set off-diagonal Laplacian components
	off_diag = 1*np.ones(N)
	#set diagonal Laplacian components
	diag = (-2)*np.ones(N)

	# Generate the diagonal and off-diagonal matrices
	A = np.vstack((off_diag, diag, off_diag))
	lap1D = scipy.sparse.dia_matrix((A,[-1,0,1]),shape=(N,N))
	speye = scipy.sparse.identity(N)
	#put diagonals together into sparse matrix format
	L = (1/(h**2))*(scipy.sparse.kron(lap1D,speye) + scipy.sparse.kron(speye,lap1D))

	#make identity matrix of same size as L
	I = scipy.sparse.identity(N**2)
	return L, I

def forward_euler_time_step(h, del_t, u, L, f, I):
	#one time step of crank-nicolson solver
	N = 1/h -1

	#u^n+1 = (I + del_t L)u^n + del_t*f
	A = (I + (del_t) * L)
	u_next = A.dot(u.flatten(order='C')) + del_t*f
	u_next = np.reshape(u_next, (N, N))

	return u_next

def forward_euler_method(h, del_t, u, f, D, x, y, plotting):

	#create sparse matrices for crank-nicolson method
	[L, I] = sparse_matrices(h)

	#calculate number of time points after 0 up to 1 (inclusive)
	Nt = int(1/del_t)

	if plotting==1:
		#set up plotting
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		# `plot_surface` expects `x` and `y` data to be 2D
		X, Y = np.meshgrid(x, y)  
		#keep z limits fixed
		ax.set_zlim(0, 1)
		plt.ion()
		#plot first frame, u(x,y,0)
		frame = ax.plot_surface(X, Y, u)
		plt.pause(0.05)

	for t in range(0,Nt):
		#solve for next u
		u = forward_euler_time_step(h, del_t, u, D*L, f[t], I)
		
		if plotting==1:
			#plot current u
			ax.collections.remove(frame)
			frame = ax.plot_surface(X, Y, u)
			plt.pause(0.05)

	return u

if __name__ == '__main__':
	# print('does nothing alone, import as a method')
	[A,I] = sparse_matrices(0.25)
	print(A.toarray())
	# setup_and_run(1/4,1/100)
	# refinement_study()
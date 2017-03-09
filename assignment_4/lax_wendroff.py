#Lax_Wendroff.py
#Carter Johnson
#MAT228B Assignment 4

#Lax-Wendroff method for solving the advection eqn
#on [0,1] with periodic BCs
#u_j^n+1 = u_j^n - v/2(u_j+1^n-u_j-1^n)+v^2/2(u_j+1^n-2u_j^n+u_j-1^n)

from __future__ import division
import numpy as np
from numpy import exp, sin, pi
from numpy.linalg import norm
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg

def LW_matrix(N, nu):
	#set sparse matrix LW for LW method
	#u_j^n+1 = (1-v^2)u_j^n + (v^2-v)/2(u_j+1^n) + (v^2+v)/2(u_j-1^n)
	#on [0,1] with periodic BCs

	#set sub-diagonal components
	sub_diag = (nu**2 +nu)/2*np.ones(N)
	#set above-diagonal components
	above_diag = (nu**2 -nu)/2*np.ones(N)
	#set diagonal components
	diag = (1-nu**2)*np.ones(N)
	#set right corner component (from periodic domain)
	right_corner = (nu**2+nu)/2*np.ones(N)
	#set left corner component (from periodic domain)
	left_corner = (nu**2-nu)/2*np.ones(N)

	# Generate the matrix
	A = np.vstack((left_corner, sub_diag, diag, above_diag, right_corner))
	LW = scipy.sparse.dia_matrix((A,[-(N-1),-1,0, 1,(N-1)]),shape=(N,N))
	return LW

def LW_method(u0, LW, Nt):
	#LW method on [0,1] w/ periodic BCs
	#with time step delT up to time Tf with number of steps Nt=Tf/delT
	#using scheme matrix LW(delX, nu)

	#start iteration with initial given u0
	u_old = u0+0

	#setup plots
	plt.axis([0, LW.shape[0], -1, 1])
	plt.ion()
	#do Nt = Tf/delT time steps
	for t in range(Nt):
		#advance u w/ LW scheme
		u_next = LW.dot(u_old)
		#update u_old
		u_old = u_next+0
		#plot
		plt.plot(u_next)
		# plt.show()
		plt.pause(0.05)
	plt.close()

	return u_next


if __name__ == '__main__':
	#system parameters
	a=1
	Tf = 1
	delX = 0.01
	Nx=90
	Nt=100
	delX =1/Nx
	delT = Tf/Nt
	nu = a*delT/delX
	print(nu)
	#make LW method
	LW = LW_matrix(Nx,nu)
	print(LW.toarray())
	#make smooth initial condition
	grid_X = [delX*i for i in range(Nx)]
	# print(grid_X)
	u0 = np.asarray([sin(2*pi*x) for x in grid_X])
	# print(u0)
	#solve advection eqn using LW method up to Tf=1
	u = LW_method(u0, LW, Nt)
	# print(u)
	print(u0-u)



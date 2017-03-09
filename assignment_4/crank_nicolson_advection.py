#Crank_Nicolson_Advection.py
#Carter Johnson
#MAT228B Assignment 4

#Crank-Nicolson analogous method for solving the advection eqn
#on [0,1] with periodic BCs
#u_j^n+1 - u_j^n + v/4(u_j+1^n - u_j-1^n) + v/4(u_j+1^n+1-u_j-1^n+1)=0

from __future__ import division
import numpy as np
from numpy import exp, sin, pi
from numpy.linalg import norm
import matplotlib.pyplot as plt
from pylab import savefig
import scipy.sparse as sparse
import scipy.sparse.linalg

def CN_matrix(N, nu):
	#set sparse matrix CN for CN-analogous method
	#u_j^n+1 + v/4(u_j+1^n+1-u_j-1^n+1) =  u_j^n - v/4(u_j+1^n - u_j-1^n)
	#on [0,1] with periodic BCs

	#set sub-diagonal components
	sub_diag = (nu/4)*np.ones(N)
	#set above-diagonal components
	above_diag = (-nu/4)*np.ones(N)
	#set diagonal components
	diag = np.ones(N)
	#set right corner component (from periodic domain)
	right_corner = (nu/4)*np.ones(N)
	#set left corner component (from periodic domain)
	left_corner = (-nu/4)*np.ones(N)

	# Generate the matrix
	A = np.vstack((left_corner, sub_diag, diag, above_diag, right_corner))
	CN = scipy.sparse.dia_matrix((A,[-(N-1),-1,0, 1,(N-1)]),shape=(N,N))
	return CN

def CN_method(u0, CN, Nt):
	#CN-analogous method on [0,1] w/ periodic BCs
	#with time step delT up to time Tf by doing Nt=Tf/delT steps
	#using scheme matrix S(delX, nu)

	#start iteration with initial given u0
	u_old = u0+0
	#setup plots
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_ylim(min(u0)-0.5, max(u0)+0.5)
	plt.ion()
	text = ax.text(0.05,0.95, r"$t=0$", transform=ax.transAxes)
	ax.plot(u_old)
	frame_no=1
	filename='CN_advection_fig0'+str(frame_no)+'.png'
	savefig(filename)
	plt.pause(0.5)

	#do Nt steps
	for t in range(Nt):
		#advance u w/ CN-type scheme
		RHS_terms = CN.dot(u_old)
		LHS_matrix = scipy.sparse.csc_matrix(CN.T)
		u_next = scipy.sparse.linalg.spsolve(LHS_matrix, RHS_terms)
		#update u_old
		u_old = u_next+0
		#plot
		ax.clear()
		text = ax.text(0.05,0.95, r"$t=%.3f$" % (t/Nt), transform=ax.transAxes)
		frame = ax.plot(u_next)
		plt.pause(0.5)
		if t%10==0:
			frame_no=frame_no+1
			if frame_no<10:
				filename='CN_advection_fig0'+str(frame_no)+'.png'
			else:
				filename='CN_advection_fig'+str(frame_no)+'.png'
			savefig(filename)

	return u_next


if __name__ == '__main__':
	#system parameters
	a=1
	Tf = 1
	delX = 0.01
	Nx = 90
	Nt = 100
	delX = 1/Nx
	delT = Tf/Nt
	nu = a*delT/delX
	print(nu)
	#make upwinding method
	CN = CN_matrix(Nx,nu)
	print(CN.toarray())
	#make smooth initial condition
	# grid_X = [delX*i for i in range(Nx)]
	# # print(grid_X)
	# u0 = np.asarray([sin(2*pi*x) for x in grid_X])

	#make disconts initial cond
	u0 = np.r_[[0 for j in range(int(Nx/4))], [1 for j in range(int(Nx/4),int(3*Nx/4))], [0 for j in range(int(3*Nx/4),Nx)] ] 

	# print(u0)
	#solve advection eqn using upwinding method up to Tf=1
	u = CN_method(u0, CN, Nt)
	# print(u)
	print(u0-u)



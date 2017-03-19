#Problem1.py
#Carter Johnson
#MAT228B Assignment 5

#Solve hyperbolic eqns
# p_t + K u_x = 0
# u_t + 1/r p_x = 0

#using Lax-Wendroff method for solving the advection eqn
#on [0,1] with ghost cell BCs

from __future__ import division
import numpy as np
from numpy import exp, sin, pi, sqrt, cos
from numpy.linalg import norm
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg
from diag_fns import make_diag_fns



def LW_matrix(N, nu):
	#set sparse matrix LW for LW method
	#u_j^n+1 = (1-v^2)u_j^n + (v^2-v)/2(u_j+1^n) + (v^2+v)/2(u_j-1^n)
	#on [0,1] ignoring BCs

	#set sub-diagonal components
	sub_diag = (nu**2 +nu)/2*np.ones(N)
	#set above-diagonal components
	above_diag = (nu**2 -nu)/2*np.ones(N)
	#set diagonal components
	diag = (1-nu**2)*np.ones(N)

	# Generate the matrix
	A = np.vstack((sub_diag, diag, above_diag))
	LW = scipy.sparse.dia_matrix((A,[-1,0, 1]),shape=(N,N))
	return LW

def acoustic_LW(u0,p0, LW1, LW2, Nt,K,r):
	#LW method on [0,1] w/ ghost cell BCs for acoustic eqn
	#with time step delT up to time Tf with number of steps Nt=Tf/delT
	#using scheme matrices LW1(delX, nu1), LW2(delX, nu2)

	diagonalize, undiagonalize = make_diag_fns(K,r)
	#start iteration with initial given u0,p0
	#diagonalize
	s_old = diagonalize(p0,u0)

	#setup plots
	plt.axis([0, LW1.shape[0]-2, -2, 2])
	plt.ion()
	#do Nt = Tf/delT time steps
	for t in range(Nt):
		#compute ghost cell components
		[p1, u1] = undiagonalize(s_old[:,0])
		# print(p1,u1)
		left_ghost = diagonalize(p1,-u1)
		# print(left_ghost)
		[pN,uN] = undiagonalize(s_old[:,-1])
		right_ghost = diagonalize((1/2)*(pN + uN*sqrt(K*r)),(1/2)*(pN/sqrt(K*r) + uN))
		# print(right_ghost)
		#add ghost cell components
		# print("s=",s_old)
		s_old = np.c_[left_ghost, s_old, right_ghost]
		# print("s=",s_old)
		#advance s w/ LW scheme separately
		s1_old = np.transpose(s_old[0,:])
		s1 = LW1.dot(s1_old)
		# print("s1=",s1)
		s2_old = np.transpose(s_old[1,:])
		s2 = LW2.dot(s2_old)
		# print("s2=",s2)
		#recombine & remove ghost cell components
		s_next = np.r_[[s1[1:-1]],[s2[1:-1]]]
		# print("s=",s_next)

		#update s_old
		s_old = s_next+0

		#plot s, p and u
		if t%5==0:
			plt.subplot(221); plt.plot(s_old[0]) ;plt.ylabel("s1")
			plt.subplot(222); plt.plot(s_old[1]); plt.ylabel("s2")
			plt.subplot(223); plt.plot(undiagonalize(s_old)[1]); plt.ylabel("u")
			plt.subplot(224); plt.plot(undiagonalize(s_old)[0]); plt.ylabel("p")
			plt.show(); plt.pause(0.5); plt.close()

	return undiagonalize(s_next)

if __name__ == '__main__':
	#system parameters
	K = 1
	r = 0.9

	#set grid spacings/time steps
	delX = 3**(-6)
	#get time step
	nu1 = 0.9
	#delT = 2**(-10)
	delT = nu1*delX/sqrt(K*r)
	#get grid points for level h
	Nx = int(round(1/delX))
	Nt = int(round(1/delT))
	X = [delX*(j-0.5) for j in range(1,Nx+1)]

	#compute advection/wave speeds
	nu1 = sqrt(K*r)*delT/delX
	print(nu1)
	nu2 = -nu1
	#make LW matrices
	LW1 = LW_matrix(Nx+2,nu1)
	LW2 = LW_matrix(Nx+2,nu2)

	#create initial condition - 0 = Square Pulses, 1= Gaussians, 2=sin/cos waves
	IC = 2
	if IC ==0:
		p0 = np.r_[np.zeros(int(Nx/4)), np.ones(int(Nx/2)), np.zeros(int(Nx/4)+1)]
		u0 = np.r_[np.zeros(int(Nx/4)), np.ones(int(Nx/2)), np.zeros(int(Nx/4)+1)]	
	if IC == 1:
		p0 = [exp(-100*(x-0.3)**2) for x in X]
		u0 = [exp(-100*(x-0.8)**2) for x in X]
	if IC==2:
		p0 = [sin(8*pi*x) for x in X]
		u0 = [cos(8*pi*x) for x in X]


	final = acoustic_LW(u0,p0,LW1,LW2,Nt,K,r)



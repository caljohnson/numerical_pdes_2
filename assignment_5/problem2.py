#Problem2.py
#Carter Johnson
#MAT228B Assignment 5

#Solve linear advection eqn
# u_t + a u_x = 0
#on [0,1] using finite volume method and flux-limiters
#with ghost cell BCs

from __future__ import division
import numpy as np
from numpy import exp, sin, pi, sqrt, cos
from numpy.linalg import norm
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg

def make_flux_function(a, delT, delX, phi):
	#constructs numerical flux function using speed a, time step delT, 
	#grid spacing delX, and flux limiter function phi


	def flux(u):
		#compute numerical flux function F_j-1/2 for j=1 to N-1
		Nx = u.shape[0]
		F = np.zeros(Nx-1)
		for j in range(1,Nx):
			#F_j-1/2 = Fup_j-1/2 + |a|/2*(1-|a delT/delX| )*delta_j-1/2
			#with delta_j-1/2 = phi(theta_j-1/2)*(u_j-u_j-1)
			#and theta_j-1/2 = (u_Jup - u_Jup-1)/(u_j - u_j-1)

			#compute Fup_j-1/2 and J_up upwind flux and jump
			if a>=0:
				flux_up = a*u[j-1]
				J_up = j-1
			else:
				flux_up = a*u[j]
				J_up = j+1
			#compute theta_j-1/2 and delta_j-1/2
			theta = (u[J_up] - u[J_up-1])/(u[j] - u[j-1])
			delta = phi(theta)*(u[j] - u[j-1])
			#compute F_j-1/2
			F[j-1] = flux_up + abs(a)/2*(1-abs(a*delT/delX))*delta
		return F

	return flux

def make_flux_limiter(n):
	#Creates the flux limiter function phi based on the integer n
	#n=0 - Upwinding phi(theta)=0
	if n==0:
		def phi(theta):
			return 0
	#n=1 - Lax-Wendroff phi(theta)=1
	if n==1:
		def phi(theta):
			return 1
	#n=2 - Beam-Warming phi(theta)=theta
	if n==2:
		def phi(theta):
			return theta	
	#n=3 - minmod phi(theta)=minmod(1,theta)
	if n==3:
		def phi(theta):
			return max(0,min(1,theta))
	#n=4 - superbee phi(theta)=max(0,min(1,2theta),min(2,theta))
	if n==4:
		def phi(theta):
			return max(0,min(1,2*theta), min(2,theta))
	#n=5 - MC phi(theta)=max(0,min((1+theta)/2,2,2theta))
	if n==5:
		def phi(theta):
			return max(0,min((1+theta)/2,2,2*theta))
	#n=6 - van Leer phi(theta)=(theta+|theta|)/(1+|theta|)
	if n==6:
		def phi(theta):
			return (theta+abs(theta))/(1+abs(theta))
	return phi		
			
def high_res_method(u0, flux, delT, delX,Tf):
	#High res methods on [0,1] w/ ghost cell periodic BCs
	#with time step delT up to time Tf with number of steps Nt=Tf/delT

	#start iteration with initial given u0
	u_old = u0

	#setup plots
	plt.axis([0, u0.shape[0], -2, 2])
	plt.ion()
	#do Nt = Tf/delT time steps
	for t in range(int(round(Tf/delT))):
		#compute ghost cell components using periodic BCs
		left_ghost = u_old[-1]
		right_ghost = u_old[0]
		#add ghost cell components
		u_full = np.r_[left_ghost, u_old, right_ghost]

		#compute flux vectors F_j+1/2, F_j-1/2
		F = flux(u_full)
		F_plus = F[1:]
		F_minus = F[0:-1]

		#compute next u
		u_next = u_old - (delT/delX)*(F_plus-F_minus)

		#update s_old
		u_old = u_next+0

		#plot s, p and u
		if t%5==0:
			plt.plot(u_old); plt.ylabel("u")
			plt.axis([0, u0.shape[0], -1, 2])
			plt.pause(0.5); plt.close()

	return u_next

if __name__ == '__main__':
	#system parameters
	a=1
	Tf=5

	#set number of grid spacings/time steps
	Nx = 90
	Nt = 500

	#get grid spacing/time step sizes
	delX = 1/Nx
	delT = 1/(Nt/Tf)

	#get courant number
	nu1 = a*delT/delX

	#create flux limiter fn phi
	n = 3 #0 Up, 1 LW, 2 BW, 3 minmod, 4 superbee, 5 MC, 6 van Leer
	phi = make_flux_limiter(n)

	#create numerical flux function
	flux = make_flux_function(a,delT,delX,phi)

	#create initial condition - 0 = wave packet, 1=smooth,low freq, 2=step function
	IC = 2
	X = [delX*(j-0.5) for j in range(1,Nx)]
	if IC ==0:
		u0 = [cos(16*pi*x)*exp(-50*(x-0.5)**2) for x in X]
	if IC == 1:
		u0 = [sin(2*pi*x)*sin(4*pi*x) for x in X]
	if IC==2:
		u0 = np.r_[np.zeros(int(Nx/4)), np.ones(int(Nx/2)), np.zeros(int(Nx/4)+1)]	

	final = high_res_method(u0,flux,delT,delX,Tf)



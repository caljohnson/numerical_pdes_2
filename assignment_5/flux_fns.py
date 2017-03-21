#Flux_fns.py
#Carter Johnson
#MAT228B Assignment 5

#Functions to compute the numerical flux vector
#using flux-limiters
#to solve linear advection eqn
# u_t + a u_x = 0
#on [0,1] using finite volume method and flux-limiters
#with ghost cell BCs
 
from __future__ import division
import numpy as np
from numpy import exp, sin, pi, sqrt
import matplotlib.pyplot as plt

def make_flux_function(a, delT, delX, phi):
	#constructs numerical flux function using speed a, time step delT, 
	#grid spacing delX, and flux limiter function phi


	def flux(u):
		#compute numerical flux function F_j-1/2 for j=1 to N-1
		Nx = u.shape[0]
		tol = 1e-10
		# print(Nx)
		F = np.zeros(Nx-3)
		# print(F.shape[0])
		for j in range(2,Nx-1):
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
			denom = u[j] - u[j-1]
			if abs(denom)<=tol:
				delta = 0
			else:	
				theta = (u[J_up] - u[J_up-1])/denom
				delta = phi(theta)*denom
			#compute F_j-1/2
			# print(j-2)
			F[j-2] = flux_up + abs(a)/2*(1-abs(a*delT/delX))*delta
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

def test_flux():
	a=1
	Nx =10
	Nt = 100
	delX = 1/Nx
	delT =1/Nt
	nu = a*delT/delX
	u0 = np.r_[np.zeros(int(Nx/4)), np.ones(int(Nx/2)), np.zeros(int(Nx/4))]
	print(u0)
	X = [delX*(j-0.5) for j in range(1,Nx)]
	#create flux limiter fn phi
	n = 0 #0 Up, 1 LW, 2 BW, 3 minmod, 4 superbee, 5 MC, 6 van Leer
	phi = make_flux_limiter(n)

	#create numerical flux function
	flux = make_flux_function(a,delT,delX,phi)
	u_old = u0+0

	#compute ghost cell components using periodic BCs
	for t in range(1,Nt+1):
		left_ghosts = np.r_[u_old[-2],u_old[-1]]
		right_ghosts = np.r_[u_old[0],u_old[1]]
		#add ghost cell components
		u_full = np.r_[left_ghosts, u_old, right_ghosts]
		print("u_full=",u_full)

		#compute flux vectors F_j+1/2, F_j-1/2
		F = flux(u_full)
		F_plus = F[1:]
		F_minus = F[0:-1]
		print("Fminus=",F_minus)
		print("F_plus =", F_plus)
		print("Ffull=",F_plus-F_minus)

		#compute next u
		u_next = u_old - (delT/delX)*(F_plus - F_minus)
		print("u=",u_next)
		input("Press Enter to continue...")
		u_old = u_next+0

	plt.plot(X, u0)
	plt.plot(X,u_next)
	plt.show()

if __name__ == '__main__':
	test_flux()
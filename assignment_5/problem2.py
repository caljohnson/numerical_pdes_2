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
from flux_fns import make_flux_function, make_flux_limiter
from tabulate import tabulate
			
def high_res_method(u0, flux, delT, delX,Tf, plots_on):
	#High res methods on [0,1] w/ ghost cell periodic BCs
	#with time step delT up to time Tf with number of steps Nt=Tf/delT

	#start iteration with initial given u0
	u_old = u0

	#setup plots
	# plt.figure(1)
	# plt.axis([0, u0.shape[0], -2, 2])
	# plt.ion()
	#do Nt = Tf/delT time steps
	Nt = int(Tf/delT)
	for t in range(1,Nt+1):
		# #compute ghost cell components using periodic BCs
		# left_ghosts = np.r_[u_old[-2],u_old[-1]]
		# right_ghosts = np.r_[u_old[0],u_old[1]]
		# #add ghost cell components
		# u_full = np.r_[left_ghosts, u_old, right_ghosts]
		u_full = np.pad(u_old, (2,2), "wrap")

		#compute flux vectors F_j+1/2, F_j-1/2
		F = flux(u_full)
		F_plus = F[1:]
		F_minus = F[0:-1]

		#compute next u
		u_next = u_old - (delT/delX)*(F_plus-F_minus)

		#update s_old
		u_old = u_next+0

		#plot s, p and u
		if t%20==0 and plots_on==1:
			plt.plot(u_old); plt.ylabel("u")
			plt.axis([0, u0.shape[0], -1, 2])
			plt.text(2,1.8,'t=%.4s' % (t*delT))
			plt.pause(0.5); plt.close()

	return u_old

def main():
	#system parameters
	a=1
	Tf=5

	#set number of grid spacings/time steps
	Nx = 450
	Nt = 500

	#get grid spacing/time step sizes
	delX = 1/Nx
	delT = 1/Nt

	#get courant number
	nu = a*delT/delX
	print(nu)

	#decide whether to plot during simulation
	plots_on = 0

	#IC choice
	IC = 2
	#create initial condition - 0 = wave packet, 1=smooth,low freq, 2=step function
	X = [delX*(j-0.5) for j in range(1,Nx+1)]
	if IC ==0:
		u0 = np.asarray([cos(16*pi*x)*exp(-50*(x-0.5)**2) for x in X])
	if IC == 1:
		u0 = np.asarray([sin(2*pi*x)*sin(4*pi*x) for x in X])
	if IC==2:
		u0=np.zeros(Nx)
		for j in range(Nx):
			if abs(X[j]-0.5)<=1/4:
				u0[j]=1
	
	final = np.zeros((7, Nx))

	#loop over flux limiters
	for n in range(7):
		#create flux limiter fn phi
		#0 Up, 1 LW, 2 BW, 3 minmod, 4 superbee, 5 MC, 6 van Leer
		phi = make_flux_limiter(n)
		
		#create numerical flux function
		flux = make_flux_function(a,delT,delX,phi)

		final[n] = high_res_method(u0,flux, delT,delX,Tf,plots_on)

	#Compare with analytic solution
	plt.figure(2)
	plt.plot(X, u0)

	#norm-errors
	errors_norm1 = np.zeros(7)
	errors_norm2 = np.zeros(7)
	errors_normmax = np.zeros(7)
	for n in range(7):
		plt.plot(X,final[n])
		#compute norm errors
		error = u0-final[n]
		errors_norm1[n] = delX*sum(abs(error))
		errors_norm2[n] = (delX*sum(error**2))**(1/2)
		errors_normmax[n] = max(abs(error))
	#display errors table
	table = [[i, errors_norm1[i], errors_norm2[i], errors_normmax[i]] for i in range(7)]
	print(tabulate(table, headers=["Method", "1-norm error", "2-norm error", "max-norm error"], tablefmt="latex"))
	
	#plt.axhline(y=.5, color='r')
	plt.text(2,1.8,'t=5')
	plt.title("Comparison of Flux Limiters")
	# plt.legend((analytic, numeric[0], numeric[1],numeric[2],numeric[3],numeric[4],numeric[5],numeric[6]),('Analytic', 'Upwinding', 'LW', 'BW', 'minmod', 'Superbee', 'MC', 'van Leer'), 'upper right')
	if IC==1:
		plt.legend(('Analytic', 'Upwinding', 'LW', 'BW', 'minmod', 'Superbee', 'MC', 'van Leer'), loc='lower right')
	else:
		plt.legend(('Analytic', 'Upwinding', 'LW', 'BW', 'minmod', 'Superbee', 'MC', 'van Leer'), loc='upper right')
	plt.text(2,1.8,'t=5')
	plt.show()


if __name__ == '__main__':
	main()
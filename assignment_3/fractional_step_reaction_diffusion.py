#Fractional_Step_Reaction_Diffusion.py
#Carter Johnson
#Mat228B Assignment 3

#Fraction Step Strang-splitting method to solve Reaction-diffusion eqn
#using Peaceman-Rachford ADI scheme on a cell-centered grid
#for solving 2-d homogeneous diffusion part with Neumann BCs
#on unit square
#and scipy ode solver for reaction terms

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import exp
from numpy.linalg import norm
from tqdm import tqdm

import scipy.sparse as sparse
import scipy.sparse.linalg
from scipy.integrate import ode
from peaceman_rachford import peaceman_rachford_method, peaceman_rachford_step, sparse_matrices

def f(t,u,a=0.1,I=0.0,eps=0.005,gamma=2):
	return [(a-u[0])*(u[0]-1)*u[0]-u[1]+I, eps*(u[0]-gamma*u[1])]

def strang_split_step(v,w, h, delT, b, L, I):
	#Strang splitting time step for fractional step method solve
	# v = b∆v + R(v,w)
	# w = R(v,w)
	N = int(1/h)-1

	#first solve diffusion on v using ADI scheme for time length ∆t/2
	v_star = peaceman_rachford_step(v,h,delT/2,b,L,I)
	#then solve reaction for ∆t using a ODE solver
	#flatten v and w from grid-lined up matrices into column vectors, put side by side
	v_and_w = np.c_[v_star.flatten(), w.flatten()]
	#solve ODE at each grid point (row of v_and_w) for one time step ∆t
	# solver = ode(f).set_integrator('vode', method='adams', order=10, rtol=0, atol=1e-6,with_jacobian=False)
	# solver.set_initial_value(v_and_w,0)
	# solver.integrate(solver.t+delT)
	# v_and_w = solver.y+0

	for i in range((N**2)):
		# solver = ode(f).set_integrator('vode', method='adams', order=10, rtol=0, atol=1e-6,with_jacobian=False)
		solver = ode(f).set_integrator('vode', method='bdf')
		solver.set_initial_value(v_and_w[i])
		solver.integrate(solver.t+delT)
		v_and_w[i] = solver.y+0

	#reshape back into v, w grid matrices
	v_starstar = np.reshape(v_and_w[:,0], (N,N))
	w_next = np.reshape(v_and_w[:,1], (N,N))

	#solve diffusion on v_starstar for ∆t/2 to get v_next
	v_next = peaceman_rachford_step(v_starstar,h,delT/2,b,L,I)

	return v_next, w_next
	

def frac_step_strang_split(h, delT, Nt, b, v_old, w_old, plotting):
	#Strang splitting fractional step method
	#solve reaction-diffusion eqn for v,w
	#up to time Nt
	N = int(1/h-1)

	#get operators
	[L,I] = sparse_matrices(h)

	if plotting==1:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		# `plot_surface` expects `x` and `y` data to be 2D
		grid_X = [h*(i-0.5) for i in range(1,N+1)]
		grid_Y = [h*(j-0.5) for j in range(1, N+1)]
		X, Y = np.meshgrid(grid_X, grid_Y)  
		#keep z limits fixed
		ax.set_zlim(0, 1)
		plt.ion()
		#plot first frame, v(x,y,0)
		frame = ax.plot_surface(X, Y, v_old)
		plt.pause(0.05)

	for t in tqdm(range(Nt)):
		#solve for next v and w
		[v_new,w_new] = strang_split_step(v_old,w_old, h, delT, b, L, I)
		
		if plotting==1 and t%(Nt/100)==0:
			#plot current v
			ax.collections.remove(frame)
			frame = ax.plot_surface(X, Y, v_new)
			plt.pause(0.05)

		v_old = v_new + 0
		w_old = w_new


	return v_new, w_new

def part_b_Run(h,delT):
	#parameters
	plotting = 1
	N = int(1/h - 1)
	Nt = 300*int(1/delT)
	a = 0.1
	gamma = 2
	eps = 0.005
	I_current=0
	D = 5*(10**(-5))

	#setup initial data
	grid_X = [h*(i-0.5) for i in range(1,N+1)]
	grid_Y = [h*(j-0.5) for j in range(1, N+1)]
	v0 = np.asarray([[exp(-100*(x**2+y**2)) for x in grid_X] for y in grid_Y])
	w0 = np.asarray([[0*x+0*y for x in grid_X] for y in grid_Y])
	
	#run Fractional Step method to solve up to time Nt
	[v,w] = frac_step_strang_split(h,delT,Nt,D, v0,w0, plotting)

def part_c_Run(h,delT):
	#parameters
	plotting = 1
	N = int(1/h - 1)
	Nt = 600*int(1/delT)
	a = 0.1
	gamma = 2
	eps = 0.005
	I_current=0
	D = 5*10**(-5)

	#setup initial data
	grid_X = [h*(i-0.5) for i in range(1,N+1)]
	grid_Y = [h*(j-0.5) for j in range(1, N+1)]
	v0 = np.asarray([[1-2*x for x in grid_X] for y in grid_Y])
	w0 = np.asarray([[0.05*y for x in grid_X] for y in grid_Y])
	
	#run Fractional Step method to solve up to time Nt
	[v,w] = frac_step_strang_split(h,delT,Nt,D, v0,w0, plotting)

if __name__ == '__main__':
	h = 2**(-4)
	delT = 2**(-4)
	part_b_Run(h,delT)
	# part_c_Run(h,delT)	

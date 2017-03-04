#Problem1.py
#Carter Johnson
#Mat228B Assignment 3

#Refinement study on the
#Peaceman-Rachford ADI scheme on a cell-centered grid
#for solving 2-d homogeneous diffusion eqn with Neumann BCs

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import exp
from numpy.linalg import norm
from tabulate import tabulate
from tqdm import tqdm
from time import clock
import scipy.sparse as sparse
import scipy.sparse.linalg

from peaceman_rachford import peaceman_rachford_method

def refinement_study():
	#refinement study for 2d diffusion using peaceman-rachford ADI scheme
	#for homogenous diffusion w/ Neumann bcs

	#set vector of grid spacings/time steps
	h = [3**(-i) for i in range(1,7)]

	#diffusion coefficient
	b = 0.1

	#Don't plot
	plotting=0

	#record successive differences + ratios, run times and runtime ratios
	diffs = np.zeros(len(h))
	diff_ratios = np.zeros(len(h))
	times = np.zeros(len(h))
	time_ratios = np.zeros(len(h))

	
	for i in tqdm(range(len(h))):
		#get time step
		delT = h[i]

		#get grid points for level h
		N = int(round(1/h[i]))
		Nt = int(round(1/delT))
		X = [h[i]*(j-0.5) for j in range(1,N+1)]
		Y = [h[i]*(j-0.5) for j in range(1, N+1)]

		#initial condition u(x,y,0)=exp(-100((x-0.3)^2+(y-0.4)^2))
		u = [[exp(-100*((x-0.3)**2+(y-0.4)**2)) for x in X] for y in Y]
		u = np.asarray(u)

		toc=clock()
		u_new = peaceman_rachford_method(h[i], delT, b, u, plotting)
		tic=clock()
		if i>0:
			# print(restriction(u_new,h[i]).shape)
			# print(u_old.shape)
			# diffs[i]=(h[i-1]**2)*norm(restriction(u_new, h[i]) - u_old,ord=1)
			diffs[i] = np.amax(restriction(u_new,h[i])-u_old)
			time_ratios[i] = (tic-toc)/times[i-1]
		if i>1:
			diff_ratios[i]=diffs[i-1]/diffs[i]
		u_old = u_new+0	
		times[i]=tic-toc


	table = [[h[i], times[i], time_ratios[i], diffs[i], diff_ratios[i]] for i in range(len(h))]
	print(tabulate(table, headers=["grid spacings/time steps", "Runtimes", "Runtime Ratios", "Successive Differences", "Difference Ratios"], tablefmt="latex"))

def restriction(u, h):
	u_f = u +0
	h2 = 3*h
	n2 = int(round(1/h2))
	u_c = np.zeros((n2, n2), dtype=float)

	#loop over coarse mesh
	for i in range(0,n2):
		for j in range(0,n2):
			u_c[i][j] = u_f[3*i+1][3*j+1]
	return u_c

if __name__ == '__main__':
	refinement_study()	
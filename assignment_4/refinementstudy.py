#Refinementstudy.py
#Carter Johnson
#Mat228B Assignment 4

#Refinement study on the
#advection eqn on [0,1] w/ periodic BCs
#for upwinding and Lax-Wendroff methods

from __future__ import division

import numpy as np
from numpy import exp,sin,pi
from numpy.linalg import norm
from tabulate import tabulate
from tqdm import tqdm
from time import clock

from upwinding import upwinding_method, upwinding_matrix
from lax_wendroff import LW_method, LW_matrix

def refinement_study_up():
	#refinement study for advection eqn on [0,1] w/ periodic BCs
	#for upwinding scheme

	#set vector of grid spacings
	h = [2**(-i) for i in range(3,17)]

	#advection speed
	a = 1
	#final time
	Tf=1

	#dont plot
	plotting=0

	#record errors + ratios, run times and runtime ratios
	errors_norm1 = np.zeros(len(h))
	norm1_ratios = np.zeros(len(h))
	errors_norm2 = np.zeros(len(h))
	norm2_ratios = np.zeros(len(h))
	errors_normmax = np.zeros(len(h))
	normmax_ratios = np.zeros(len(h))
	times = np.zeros(len(h))
	time_ratios = np.zeros(len(h))

	
	for i in tqdm(range(len(h))):
		#get time step
		delT = 0.9*a*h[i]
		#get Courant number
		nu = a*delT/h[i]

		#make upwinding method
		S = upwinding_matrix(h[i],nu)
		# print(S.toarray())
		#make smooth initial condition
		N = int(round(1/h[i]))
		grid_X = [h[i]*j for j in range(N)]
		u0 = np.asarray([sin(2*pi*x) for x in grid_X])

		toc=clock()
		u = upwinding_method(u0, S, delT, Tf)
		tic=clock()
		error = u0-u
		errors_norm1[i] = h[i]*sum(abs(error))
		errors_norm2[i] = (h[i]*sum(error**2))**(1/2)
		errors_normmax[i] = max(abs(error))
		times[i]=tic-toc
		if i>0:
			norm1_ratios[i] = errors_norm1[i-1]/errors_norm1[i]
			norm2_ratios[i] = errors_norm2[i-1]/errors_norm2[i]
			normmax_ratios[i] = errors_norm2[i-1]/errors_normmax[i]
			time_ratios[i] = (tic-toc)/times[i-1]

	table = [[h[i], errors_norm1[i], errors_norm2[i], errors_normmax[i], times[i], time_ratios[i]] for i in range(len(h))]
	print(tabulate(table, headers=["delta x", "1-norm error", "2-norm error", "max-norm error", "runtime", "runtime ratios"], tablefmt="latex"))
	norm1_table = [[h[i], errors_norm1[i], norm1_ratios[i]] for i in range(len(h))]
	print(tabulate(norm1_table, headers=["delta x", "1-norm error", "Ratios"], tablefmt="latex"))
	norm2_table = [[h[i], errors_norm2[i], norm2_ratios[i]] for i in range(len(h))]
	print(tabulate(norm2_table, headers=["delta x", "2-norm error", "Ratios"], tablefmt="latex"))
	normmax_table = [[h[i], errors_normmax[i], normmax_ratios[i]] for i in range(len(h))]
	print(tabulate(normmax_table, headers=["delta x", "Max-norm error", "Ratios"], tablefmt="latex"))
	time_table = [[h[i], times[i], time_ratios[i]] for i in range(len(h))]
	print(tabulate(time_table, headers=["delta x", "Runtimes", "Runtime Ratios"], tablefmt="latex"))

def refinement_study_LW():
	#refinement study for advection eqn on [0,1] w/ periodic BCs
	#for Lax-Wendroff scheme

	#set vector of grid spacings
	h = [2**(-i) for i in range(2,16)]

	#advection speed
	a = 1
	#final time
	Tf=1

	#dont plot
	plotting=0

	#record errors + ratios, run times and runtime ratios
	errors_norm1 = np.zeros(len(h))
	norm1_ratios = np.zeros(len(h))
	errors_norm2 = np.zeros(len(h))
	norm2_ratios = np.zeros(len(h))
	errors_normmax = np.zeros(len(h))
	normmax_ratios = np.zeros(len(h))
	times = np.zeros(len(h))
	time_ratios = np.zeros(len(h))

	
	for i in tqdm(range(len(h))):
		#get time step
		delT = 0.9*a*h[i]
		#get Courant number
		nu = a*delT/h[i]

		#make LW method
		LW = LW_matrix(h[i],nu)
		# print(S.toarray())
		#make smooth initial condition
		N = int(round(1/h[i]))
		grid_X = [h[i]*j for j in range(N)]
		u0 = np.asarray([sin(2*pi*x) for x in grid_X])

		toc=clock()
		u = LW_method(u0, LW, delT, Tf)
		tic=clock()
		error = u0-u
		errors_norm1[i] = h[i]*sum(abs(error))
		errors_norm2[i] = (h[i]*sum(error**2))**(1/2)
		errors_normmax[i] = max(abs(error))
		times[i]=tic-toc
		if i>0:
			norm1_ratios[i] = errors_norm1[i-1]/errors_norm1[i]
			norm2_ratios[i] = errors_norm2[i-1]/errors_norm2[i]
			normmax_ratios[i] = errors_norm2[i-1]/errors_normmax[i]
			time_ratios[i] = (tic-toc)/times[i-1]

	table = [[h[i], errors_norm1[i], errors_norm2[i], errors_normmax[i], times[i], time_ratios[i]] for i in range(len(h))]
	print(tabulate(table, headers=["delta x", "1-norm error", "2-norm error", "max-norm error", "runtime", "runtime ratios"], tablefmt="latex"))
	norm1_table = [[h[i], errors_norm1[i], norm1_ratios[i]] for i in range(len(h))]
	print(tabulate(norm1_table, headers=["delta x", "1-norm error", "Ratios"], tablefmt="latex"))
	norm2_table = [[h[i], errors_norm2[i], norm2_ratios[i]] for i in range(len(h))]
	print(tabulate(norm2_table, headers=["delta x", "2-norm error", "Ratios"], tablefmt="latex"))
	normmax_table = [[h[i], errors_normmax[i], normmax_ratios[i]] for i in range(len(h))]
	print(tabulate(normmax_table, headers=["delta x", "Max-norm error", "Ratios"], tablefmt="latex"))
	time_table = [[h[i], times[i], time_ratios[i]] for i in range(len(h))]
	print(tabulate(time_table, headers=["delta x", "Runtimes", "Runtime Ratios"], tablefmt="latex"))

if __name__ == '__main__':
	# refinement_study_up()
	refinement_study_LW()	
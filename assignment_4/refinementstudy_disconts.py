#Refinementstudy.py
#Carter Johnson
#Mat228B Assignment 4

#Refinement study on the
#advection eqn on [0,1] w/ periodic BCs
#for upwinding and Lax-Wendroff methods
#for discontinous IC

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

	#set vector of number of grid points Nx
	Nx = [90*(2**i) for i in range(0,10)]

	#advection speed
	a = 1
	#final time
	Tf=1

	#dont plot
	plotting=0

	#record errors + ratios, run times and runtime ratios
	errors_norm1 = np.zeros(len(Nx))
	norm1_ratios = np.zeros(len(Nx))
	errors_norm2 = np.zeros(len(Nx))
	norm2_ratios = np.zeros(len(Nx))
	errors_normmax = np.zeros(len(Nx))
	normmax_ratios = np.zeros(len(Nx))
	times = np.zeros(len(Nx))
	time_ratios = np.zeros(len(Nx))

	
	for i in tqdm(range(len(Nx))):
		#get delX
		delX = 1/Nx[i]
		#get number of time steps and time step
		Nt = int(Nx[i]/0.9)
		delT = Tf/Nt
		#get Courant number
		nu = a*delT/delX

		#make upwinding method
		S = upwinding_matrix(Nx[i],nu)
		# print(S.toarray())
		#make discontinuou initial condition
		u0 = np.r_[[0 for j in range(int(Nx[i]/4))], [1 for j in range(int(Nx[i]/4),int(3*Nx[i]/4))], [0 for j in range(int(3*Nx[i]/4),Nx[i])] ] 

		toc=clock()
		u = upwinding_method(u0, S, Nt)
		tic=clock()
		error = u0-u
		errors_norm1[i] = delX*sum(abs(error))
		errors_norm2[i] = (delX*sum(error**2))**(1/2)
		errors_normmax[i] = max(abs(error))
		times[i]=tic-toc
		if i>0:
			norm1_ratios[i] = errors_norm1[i-1]/errors_norm1[i]
			norm2_ratios[i] = errors_norm2[i-1]/errors_norm2[i]
			normmax_ratios[i] = errors_norm2[i-1]/errors_normmax[i]
			time_ratios[i] = (tic-toc)/times[i-1]

	table = [[Nx[i], errors_norm1[i], errors_norm2[i], errors_normmax[i], times[i], time_ratios[i]] for i in range(len(Nx))]
	print(tabulate(table, headers=["Nx", "1-norm error", "2-norm error", "max-norm error", "runtime", "runtime ratios"], tablefmt="latex"))
	norm1_table = [[Nx[i], errors_norm1[i], norm1_ratios[i]] for i in range(len(Nx))]
	print(tabulate(norm1_table, headers=["Nx", "1-norm error", "Ratios"], tablefmt="latex"))
	norm2_table = [[Nx[i], errors_norm2[i], norm2_ratios[i]] for i in range(len(Nx))]
	print(tabulate(norm2_table, headers=["Nx", "2-norm error", "Ratios"], tablefmt="latex"))
	normmax_table = [[Nx[i], errors_normmax[i], normmax_ratios[i]] for i in range(len(Nx))]
	print(tabulate(normmax_table, headers=["Nx", "Max-norm error", "Ratios"], tablefmt="latex"))
	time_table = [[Nx[i], times[i], time_ratios[i]] for i in range(len(Nx))]
	print(tabulate(time_table, headers=["Nx", "Runtimes", "Runtime Ratios"], tablefmt="latex"))

def refinement_study_LW():
	#refinement study for advection eqn on [0,1] w/ periodic BCs
	#for Lax-Wendroff scheme

	#set vector of number of grid points Nx
	Nx = [90*(2**i) for i in range(0,10)]

	#advection speed
	a = 1
	#final time
	Tf=1

	#dont plot
	plotting=0

	#record errors + ratios, run times and runtime ratios
	errors_norm1 = np.zeros(len(Nx))
	norm1_ratios = np.zeros(len(Nx))
	errors_norm2 = np.zeros(len(Nx))
	norm2_ratios = np.zeros(len(Nx))
	errors_normmax = np.zeros(len(Nx))
	normmax_ratios = np.zeros(len(Nx))
	times = np.zeros(len(Nx))
	time_ratios = np.zeros(len(Nx))

	
	for i in tqdm(range(len(Nx))):
		#get delX
		delX = 1/Nx[i]
		#get number of time steps and time step
		Nt = int(Nx[i]/0.9)
		delT = Tf/Nt
		#get Courant number
		nu = a*delT/delX

		#make LW method
		LW = LW_matrix(Nx[i],nu)
		
		#make discontinuou initial condition
		u0 = np.r_[[0 for j in range(int(Nx[i]/4))], [1 for j in range(int(Nx[i]/4),int(3*Nx[i]/4))], [0 for j in range(int(3*Nx[i]/4),Nx[i])] ] 


		toc=clock()
		u = LW_method(u0, LW, Nt)
		tic=clock()
		error = u0-u
		errors_norm1[i] = delX*sum(abs(error))
		errors_norm2[i] = (delX*sum(error**2))**(1/2)
		errors_normmax[i] = max(abs(error))
		times[i]=tic-toc
		if i>0:
			norm1_ratios[i] = errors_norm1[i-1]/errors_norm1[i]
			norm2_ratios[i] = errors_norm2[i-1]/errors_norm2[i]
			normmax_ratios[i] = errors_norm2[i-1]/errors_normmax[i]
			time_ratios[i] = (tic-toc)/times[i-1]

	table = [[Nx[i], errors_norm1[i], errors_norm2[i], errors_normmax[i], times[i], time_ratios[i]] for i in range(len(Nx))]
	print(tabulate(table, headers=["Nx", "1-norm error", "2-norm error", "max-norm error", "runtime", "runtime ratios"], tablefmt="latex"))
	norm1_table = [[Nx[i], errors_norm1[i], norm1_ratios[i]] for i in range(len(Nx))]
	print(tabulate(norm1_table, headers=["Nx", "1-norm error", "Ratios"], tablefmt="latex"))
	norm2_table = [[Nx[i], errors_norm2[i], norm2_ratios[i]] for i in range(len(Nx))]
	print(tabulate(norm2_table, headers=["Nx", "2-norm error", "Ratios"], tablefmt="latex"))
	normmax_table = [[Nx[i], errors_normmax[i], normmax_ratios[i]] for i in range(len(Nx))]
	print(tabulate(normmax_table, headers=["Nx", "Max-norm error", "Ratios"], tablefmt="latex"))
	time_table = [[Nx[i], times[i], time_ratios[i]] for i in range(len(Nx))]
	print(tabulate(time_table, headers=["Nx", "Runtimes", "Runtime Ratios"], tablefmt="latex"))

if __name__ == '__main__':
 	refinement_study_up()
 	refinement_study_LW()	
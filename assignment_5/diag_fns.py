#Diag_fns.py
#Carter Johnson
#MAT228B Assignment 5

#Diagonalizing functions for converting hyperbolic eqns
# p_t + K u_x = 0
# u_t + 1/r p_x = 0
#into separated advection eqns
# s1_t + sqrt(K/r) s1_x = 0
# s2_t - sqrt(K/r) s2_x = 0
 
from __future__ import division
import numpy as np
from numpy import exp, sin, pi, sqrt

def make_diag_fns(K,r):

	def diagonalize(p, u):
		#puts p and u into diagonalized e-vector coordinates s1, s2
		#accepts p and u as row vectors and stacks them in a matrix
		v = np.asarray([p,u])
		Winv = np.asarray([[sqrt(K*r), sqrt(K*r)],[1,-1]])
		return Winv.dot(v)

	def undiagonalize(s):
		#puts e-vector coordinates s1, s2 into original p,u
		W = np.asarray([[1/(2*sqrt(K*r)), 1/2],[1/(2*sqrt(K*r)),-1/2]])
		return W.dot(s)

	return diagonalize, undiagonalize

def test_diags():
	#test make diag fns
	p = [1,10]
	u = [2,10]
	K = 1
	r = 2
	diagonalize,undiagonalize = make_diag_fns(K,r)
	s = diagonalize(p,u)
	[p,u] = undiagonalize(s)
	print(s)
	print(p)
	print(u)

if __name__ == '__main__':
	test_diags()
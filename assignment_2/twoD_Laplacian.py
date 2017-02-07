#twoD_Laplacian.py
#Carter Johnson

#MAT 228B
#Assignment 2

#Functions to create a 2-D Laplacian for spacing h
#and to apply the Laplacian to a solution matrix flattened to a vector

from __future__ import division

import numpy as np
from math import exp, sin, pi

import scipy.sparse as sparse
import scipy.sparse.linalg

def apply_Laplacian(u, h, A):
	#remove Dirichlet BC's from u to work with Au
	#and flatten the matrix into a vector for Au multiplication
	u_inner = u[1:-1, 1:-1].flatten(order='C')
	# A = get_Laplacian(h)

	#apply Laplacian to get matrix-vector product
	product = A.dot(u_inner)

	#shape matrix-vector product back into matrix with Dirichlet BC padding
	n = int(1/h)-1
	product_matrix = np.reshape(product, (n, n))
	padded_prod_matrix = np.pad(product_matrix, ((1,1),(1,1)), mode='constant')

	return padded_prod_matrix

def get_Laplacian(h):
    N = int(1/h)-1

    # Define diagonals
    off_diag = 1*np.ones(N)
    diag = (-2)*np.ones(N)

    # Generate the diagonal and off-diagonal matrices
    A = np.vstack((off_diag,diag,off_diag))
    lap1D = scipy.sparse.dia_matrix((A,[-1,0,1]),shape=(N,N))
    speye = scipy.sparse.identity(N)
    L = (1/(h**2))*(scipy.sparse.kron(lap1D,speye) + scipy.sparse.kron(speye,lap1D))
    return L
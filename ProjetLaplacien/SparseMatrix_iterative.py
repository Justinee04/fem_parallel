#!/bin/env python
# -*- coding: utf-8 -*-
import mesh
import fem
import fem_laplacian as laplacian
import splitter
from math import cos,sin,pi,sqrt
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import visu_split_mesh as VSM
import visu_solution as VS
from conjugate_gradient import *

def g(x,y) :
    return cos(2*pi*x)+sin(2*pi*y)

m,cl = mesh.read("CarreMedium.msh")
coords    = m.vertices
elt2verts = m.elt2verts
nbVerts = coords.shape[0]
nbElts  = elt2verts.shape[0]
print('nbVerts : {}'.format(nbVerts))
print('nbElts  : {}'.format(nbElts))
begVert2Elts, vert2elts = m.comp_vertices_to_elements()

begRows, indCols = fem.compute_skeleton_sparse_matrix(elt2verts, (begVert2Elts, vert2elts) )
nz = begRows[-1]
print("Number of non zero in sparse matrix : {}".format(nz))

spCoefs = np.zeros( (nz,), np.double)
for iElt in range(nbElts):
    iVertices = elt2verts[iElt,:]
    crd1 = coords[iVertices[0],:]
    crd2 = coords[iVertices[1],:]
    crd3 = coords[iVertices[2],:]
    matElem = laplacian.compute_elementary_matrix(crd1, crd2, crd3)
    fem.add_elementary_matrix_to_csr_matrix((begRows,indCols,spCoefs), (iVertices, iVertices, matElem))
 
# Assemblage second membre :
f = np.zeros(nbVerts, np.double)
for iVert in range(nbVerts):
    if ( cl[iVert] > 0 ) :
        f[iVert] += g(coords[iVert,0],coords[iVert,1])
b = np.zeros(nbVerts, np.double)
for i in range(nbVerts) :
    for ptR in range(begRows[i],begRows[i+1]):
        b[i] -= spCoefs[ptR]*f[indCols[ptR]]        
# Il faut maintenant tenir compte des conditions limites :
for iVert in range(nbVerts):
    if cl[iVert] > 0: # C'est une condition limite !
        # Suppression de la ligne avec 1 sur la diagonale :
        for i in range(begRows[iVert],begRows[iVert+1]):
            if indCols[i] != iVert :
                spCoefs[i] = 0.
            else :
                spCoefs[i] = 1.
        # Suppression des coefficients se trouvant sur la colonne iVert :
        for iRow in range(nbVerts):
            if iRow != iVert :
                for ptCol in range(begRows[iRow],begRows[iRow+1]):
                    if indCols[ptCol] == iVert :
                        spCoefs[ptCol] = 0.
                        
        b[iVert] = f[iVert]
# On definit ensuite la matrice :
spMatrix = sparse.csc_matrix((spCoefs, indCols, begRows),
                             shape=(nbVerts,nbVerts))
#print("Matrice creuse : {}".format(spMatrix))

# Visualisation second membre :
VS.view( coords, elt2verts, b, title = "second membre" )

# Résolution du problème séquentiel avec gradient conjugue :
sol, info = solver_gc(spMatrix, b, None, tol=1.E-14, M=None, verbose=True)
print(f"||r_final|| = {info[0]}, nombre itérations : {info[1]}")
#print(sol)
d = spMatrix.dot(sol) - b
print("||A.x-b||/||b|| = {}".format(sqrt(d.dot(d)/b.dot(b))))
# Visualisation de la solution :
VS.view( coords, elt2verts, sol, title = "Solution" )

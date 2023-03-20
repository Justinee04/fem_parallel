import numpy as np

from splitter import *
from mpi4py import MPI
import visu_split_mesh

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()


def get_nodes_subdomain(sub_domain, rank):
    print(f"Domaine {rank} : nombre d'éléments locaux {sub_domain.shape[0]}")
    print(f"Indices globaux des éléments : {sub_domain}")
    s = set()
    for glob_el in sub_domain:
        s = s | set(msh.elt2verts[glob_el].tolist())
    print(s)
    connec_mat = np.array(list(s))
    print("connec_mat", connec_mat)
    print(np.in1d(connec_mat, np.array([1, 3, 5])).nonzero()[0])
    return s


if __name__ == '__main__':
    import mesh
    import visu_split_mesh as VSM

    print("This is process ", rank, " of ", size, " processes.")
    nb_domains = size

    print("Dissection du domaine à partir de ses éléments")

    msh, cl = mesh.read("CarrePetit.msh")
    vert2elts = msh.comp_vertices_to_elements()

    splitted_elt = split_element_mesh(nb_domains, msh.elt2verts, msh.vertices)

    set.intersection(*[set(x) for x in splitted_elt])

    # Récupération des noeuds communs entre les sous-domaines
    interf_one2one = [get_nodes_subdomain(splitted_elt[rank], rank) & get_nodes_subdomain(splitted_elt[k], k) for k in
                      range(nb_domains) if k != rank]
    print("nodes" ,interf_one2one[0])

    interfNodes = list(set().union(*interf_one2one))

    # Elements globaux du sous-domaine de rang rank
    el_subdomain = splitted_elt[rank]
    # Sommets globaux du sous-domaine de rang rank
    verts = list(set(msh.elt2verts[list(el_subdomain)].flatten()))
    verts.sort()
    print("verts", verts)
    # L'index est le numero local du sommet et la valeur est le numero global du sommet
    loc2glob_array = np.array(verts)
    # Noeud a l'interface en numerotation locale !!
    interf_loc = np.in1d(verts, list(interf_one2one[0])).nonzero()[0]
    print("loc ========", interf_loc)


    def w(x):
        return np.where(loc2glob_array == x)[0][0]


    coord_som_loc = msh.vertices[verts]
    el2vert_loc = np.vectorize(w)(msh.elt2verts[list(el_subdomain)])
    ##print("coord_som_loc ===",coord_som_loc)
    visu_split_mesh.view(coord_som_loc, el2vert_loc, 1,
                         np.zeros(len(verts)), indInterfNodes=interf_loc, title='Partition par éléments ' + str(rank))

    import mesh
    import fem
    import fem_laplacian as laplacian
    import splitter
    from math import cos, sin, pi, sqrt
    from scipy import sparse
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    import visu_split_mesh as VSM
    import visu_solution as VS
    from conjugate_gradient import *


    def g(x, y):
        return cos(2 * pi * x) + sin(2 * pi * y)


    m = mesh.Mesh(coord_som_loc, el2vert_loc)
    print("cl av======", cl)
    cl = cl[loc2glob_array]
    print("cl ap======", cl)
    coords = m.vertices
    elt2verts = m.elt2verts
    nbVerts = coords.shape[0]
    nbElts = elt2verts.shape[0]
    print('nbVerts : {}'.format(nbVerts))
    print('nbElts  : {}'.format(nbElts))
    begVert2Elts, vert2elts = m.comp_vertices_to_elements()

    begRows, indCols = fem.compute_skeleton_sparse_matrix(elt2verts, (begVert2Elts, vert2elts))
    nz = begRows[-1]
    print("Number of non zero in sparse matrix : {}".format(nz))

    spCoefs = np.zeros((nz,), np.double)
    for iElt in range(nbElts):
        iVertices = elt2verts[iElt, :]
        crd1 = coords[iVertices[0], :]
        crd2 = coords[iVertices[1], :]
        crd3 = coords[iVertices[2], :]
        matElem = laplacian.compute_elementary_matrix(crd1, crd2, crd3)
        fem.add_elementary_matrix_to_csr_matrix((begRows, indCols, spCoefs), (iVertices, iVertices, matElem))

    # Assemblage second membre :
    f = np.zeros(nbVerts, np.double)
    for iVert in range(nbVerts):
        if (cl[iVert] > 0):
            f[iVert] += g(coords[iVert, 0], coords[iVert, 1])
    b = np.zeros(nbVerts, np.double)
    for i in range(nbVerts):
        for ptR in range(begRows[i], begRows[i + 1]):
            b[i] -= spCoefs[ptR] * f[indCols[ptR]]
            # Il faut maintenant tenir compte des conditions limites :
    for iVert in range(nbVerts):
        if cl[iVert] > 0:  # C'est une condition limite !
            # Suppression de la ligne avec 1 sur la diagonale :
            for i in range(begRows[iVert], begRows[iVert + 1]):
                if indCols[i] != iVert:
                    spCoefs[i] = 0.
                else:
                    spCoefs[i] = 1.
            # Suppression des coefficients se trouvant sur la colonne iVert :
            for iRow in range(nbVerts):
                if iRow != iVert:
                    for ptCol in range(begRows[iRow], begRows[iRow + 1]):
                        if indCols[ptCol] == iVert:
                            spCoefs[ptCol] = 0.
            b[iVert] = f[iVert]

    comm.send(b[interf_loc], dest=(rank + 1) % 2, tag=1)
    comm.send(interf_one2one,dest=(rank + 1) % 2, tag=0)
    b_interf = comm.recv(source=(rank + 1) % 2, tag=1)
    numglob_interf_other_proc = comm.recv(source=(rank + 1) % 2, tag=0)
    numloc_interf_other_proc = np.in1d(verts, list(numglob_interf_other_proc[0])).nonzero()[0]
    for i in interf_loc:
        j = numloc_interf_other_proc.index(i)
        b[i] += b_interf[j]

    for iVert in range(nbVerts):
        if (cl[iVert] > 0) and iVert in interf_loc :
            b[iVert] = f[iVert]


    # On definit ensuite la matrice :
    spMatrix = sparse.csc_matrix((spCoefs, indCols, begRows),
                                 shape=(nbVerts, nbVerts))
    print("Matrice creuse : {}".format(spMatrix))

    # Visualisation second membre :
    VS.view(coords, elt2verts, b, title="second membre")

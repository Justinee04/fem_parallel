import numpy as np

from splitter import *
from mpi4py import MPI
import visu_split_mesh

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()


def splitter_parallel(sub_domain, rank):
    print(f"Domaine {rank} : nombre d'éléments locaux {sub_domain.shape[0]}")
    print(f"Indices globaux des éléments : {sub_domain}")
    s = set()
    for glob_el in sub_domain:
        s = s | set(msh.elt2verts[glob_el].tolist())
    print(s)
    connec_mat = np.array(list(s))
    print(connec_mat)
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

    noeuds_sub = splitter_parallel(splitted_elt[rank], rank)
    set.intersection(*[set(x) for x in splitted_elt])

    interf_one2one = [splitter_parallel(splitted_elt[rank], rank) & splitter_parallel(splitted_elt[k], k) for k in
                      range(nb_domains) if k != rank]
    interfNodes = list(set().union(*interf_one2one))

    el_subdomain = splitted_elt[rank]
    verts = list(set(msh.elt2verts[list(el_subdomain)].flatten()))
    verts.sort()


    def w(x):
        return np.where(np.array(verts) == x)[0][0]


    visu_split_mesh.view(msh.vertices[verts], np.vectorize(w)(msh.elt2verts[list(el_subdomain)]), 1,
                         np.zeros(len(verts)),title='Partition par éléments '+str(rank))

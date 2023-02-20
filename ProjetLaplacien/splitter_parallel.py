from splitter import *

from mpi4py import MPI

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()


def splitter_parallel(sub_domain):
    print(f"Domaine {i} : nombre d'éléments locaux {sub_domain.shape[0]}")
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
    nb_domains = 4


    print("Dissection du domaine à partir de ses éléments")
    splitted_elt = split_element_mesh(nb_domains, msh.elt2verts, msh.vertices)

    splitter_parallel(splitted_elt[rank])

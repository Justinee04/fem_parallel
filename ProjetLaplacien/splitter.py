# Découper par noeud ou par élément un maillage triangulaire
from hilbertcurve import HilbertCurve
import numpy as np

OX = 0
OY = 1


def mid_point(p1, p2):
    return (0.5 * (p1[OX] + p2[OX]), 0.5 * (p1[OY] + p2[OY]))


def compute_bounding_box(points: np.array):
    assert (len(np.shape(points)) == 2)

    crd_min = (np.min(points[:, OX]), np.min(points[:, OY]))
    crd_max = (np.max(points[:, OX]), np.max(points[:, OY]))

    return [crd_min, crd_max]


def compute_morton_ordering(vertices: np.array, bbox, N: int) -> np.array:
    hilbert_curve = HilbertCurve(N, 2)
    pN: int = (1 << N) - 1
    lgth = [bbox[1][OX] - bbox[0][OX], bbox[1][OY] - bbox[0][OY]]
    return np.array([[iVert,
                      hilbert_curve.distance_from_point([int(pN * (vertices[iVert, OX] - bbox[0][OX]) / lgth[OX]),
                                                         int(pN * (vertices[iVert, OY] - bbox[0][OY]) / lgth[OY])])]
                     for iVert in range(vertices.shape[0])])


def split_node_mesh(nb_subdomains: int, vertices: np.array):
    """
    Usage : splitter.split_node_mesh( nb_domains, vertices )
    où nb_domains est le nombre de sous-domaines décomposant le domaine initial,
    et vertices les coordonnées des sommets du maillage à découper stockée par point
    """
    bbox = compute_bounding_box(vertices)
    morton_numbering = compute_morton_ordering(vertices, bbox, 20)
    sort_indices = np.argsort(morton_numbering[:, 1])
    morton_numbering = morton_numbering[sort_indices, :]
    nb_glob_verts: int = vertices.shape[0]
    nb_loc_verts: int = nb_glob_verts // nb_subdomains
    nb_suppl_verts = nb_glob_verts % nb_subdomains
    splitted_vertices = []
    start_indices: int = 0
    for iDom in range(nb_subdomains):
        nb_verts = nb_loc_verts + (1 if iDom < nb_suppl_verts else 0)
        splitted_vertices.append(np.array(morton_numbering[start_indices:start_indices + nb_verts, 0]))
        start_indices += nb_verts
    return splitted_vertices


def split_element_mesh(nb_domains: int, elt2vertices: np.array, vertices: np.array):
    """
    Usage : splitter.split_element_mesh( nbDoms, el2vertices, vertices )
    où nbDoms est le nombre de sous-domaine découpant le domaine initial,
    elt2vertices la connectivité donnant à chaque triangle l'indice des sommets correspondant (dans le sens direct),
    vertices les coordonnées des sommets stockées par point
    """
    nbVerts: int = vertices.shape[0]
    nbElts: int = elt2vertices.shape[0]
    bbox = compute_bounding_box(vertices)
    # Calcul du barycentre de chaque triangle du maillage :
    bary_coords = np.array([(vertices[elt2vertices[iElt, 0], :] + vertices[elt2vertices[iElt, 1], :] + vertices[
                                                                                                       elt2vertices[
                                                                                                           iElt, 2],
                                                                                                       :]) / 3.
                            for iElt in range(nbElts)])
    morton_numbering = compute_morton_ordering(bary_coords, bbox, 20)
    sort_indices = np.argsort(morton_numbering[:, 1])
    morton_numbering = morton_numbering[sort_indices, :]
    nb_loc_elts: int = nbElts // nb_domains
    nb_suppl_elts = nbElts % nb_domains
    splitted_elements = []
    start_indices: int = 0
    for iDom in range(nb_domains):
        nb_elts = nb_loc_elts + (1 if iDom < nb_suppl_elts else 0)
        splitted_elements.append(np.array(morton_numbering[start_indices:start_indices + nb_elts, 0]))
        start_indices += nb_elts
    return splitted_elements


if __name__ == '__main__':
    import mesh
    import visu_split_mesh as VSM

    nb_domains = 4
    msh, cl = mesh.read("CarrePetit.msh")
    vert2elts = msh.comp_vertices_to_elements()
    print("Dissection du domaine à partir de ses sommets")
    splitted_node = split_node_mesh(nb_domains, msh.vertices)
    i = 0
    for a in splitted_node:
        print(f"Domaine {i} : nombre de sommets locaux {a.shape[0]}")
        print(f"\tIndices globaux des sommets : {a}")
        i += 1

    nbVerts = msh.vertices.shape[0]
    print(f"nbVerts : {nbVerts}")
    vert2dom = np.zeros((nbVerts,), np.double)
    ia = 0.
    for a in splitted_node:
        for v in a:
            vert2dom[v] = ia
        ia += 1

    mask = np.zeros((nbVerts,), np.short)
    for e in msh.elt2verts:
        d1 = vert2dom[e[0]]
        d2 = vert2dom[e[1]]
        d3 = vert2dom[e[2]]
        if (d1 != d2) or (d1 != d3) or (d2 != d3):
            mask[e[0]] = 1
            mask[e[1]] = 1
            mask[e[2]] = 1

    nbInterf = 0
    for m in mask:
        if m == 1:
            nbInterf += 1
    interfNodes = np.empty(nbInterf, np.int64)
    nbInterf = 0
    for im in range(mask.shape[0]):
        if mask[im] == 1:
            interfNodes[nbInterf] = im
            nbInterf += 1

    VSM.view(msh.vertices, msh.elt2verts, nb_domains, vert2dom, indInterfNodes=interfNodes,
             title='Partition par sommets')

    print("Dissection du domaine à partir de ses éléments")
    splitted_elt = split_element_mesh(nb_domains, msh.elt2verts, msh.vertices)
    i = 0
    for a in splitted_elt:
        print(f"Domaine {i} : nombre d'éléments locaux {a.shape[0]}")
        print(f"Indices globaux des éléments : {a}")
        s = set()
        for glob_el in a:
            s = s | set(msh.elt2verts[glob_el].tolist())
        print(s)
        i += 1

    nbElts = msh.elt2verts.shape[0]
    elt2doms = np.zeros((nbElts,), np.double)
    ia = 0.
    for a in splitted_elt:
        for e in a:
            elt2doms[e] = ia
        ia += 1

    # Calcul l'interface :
    ie = 0
    mask = np.array([-1, ] * nbVerts, np.short)
    for e in msh.elt2verts:
        d = elt2doms[ie]
        if mask[e[0]] == -1:
            mask[e[0]] = d
        elif mask[e[0]] != d:
            mask[e[0]] = -2
        if mask[e[1]] == -1:
            mask[e[1]] = d
        elif mask[e[1]] != d:
            mask[e[1]] = -2
        if mask[e[2]] == -1:
            mask[e[2]] = d
        elif mask[e[2]] != d:
            mask[e[2]] = -2
        ie += 1

    # A paralléliser !
    nbInterf = 0
    for m in mask:
        if m == -2:
            nbInterf += 1

    interfNodes = np.empty(nbInterf, np.int64)
    nbInterf = 0
    for im in range(mask.shape[0]):
        if mask[im] == -2:
            interfNodes[nbInterf] = im
            nbInterf += 1

    # Jusque la

    VSM.view(msh.vertices, msh.elt2verts, nb_domains, elt2doms, indInterfNodes=interfNodes,
             title='Partition par elements')

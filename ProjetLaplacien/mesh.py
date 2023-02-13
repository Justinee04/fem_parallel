# Lecture et gestion des maillages de type gmsh
import numpy as np
import re

class Mesh:
    def __init__( self, vertices : np.array, elt2verts : np.array ):
        """
        Usage : m = Mesh( vertices, elt2verts )
        Créer un maillage en 2D ayant pour sommets vertices de forme (nombre de sommets, 2)
        et de connectivité elt2vert (de forme (nombre d'éléments, 3))
        """
        self.vertices = vertices 
        self.elt2verts= elt2verts
    
    def comp_vertices_to_elements( self, glob2loc : np.array = None):
        """
        Usage : (begVert2Elts, vert2elts) = m.vertices_to_elements( glob2loc = None)
        Calcule la connectivité inverse sommets vers éléments sous un format morse.
        """
        nb_elements = self.elt2verts.shape[0]
        nb_vertices = self.vertices.shape[0]
        if glob2loc is None :
            g2l = np.arange(nb_vertices, dtype=np.int64)
        else:
            loc_nodes = g2l[g2l != -1]
            nb_vertices = loc_nodes.shape[0]
            g2l = glob2loc 
        l_vertices2elements = [ [] for _ in range(nb_vertices)]
        for iElt in range(nb_elements):
            v1 : int = self.elt2verts[iElt,0]
            v2 : int = self.elt2verts[iElt,1]
            v3 : int = self.elt2verts[iElt,2]

            if g2l[v1] >= 0: l_vertices2elements[g2l[v1]].append(iElt)
            if g2l[v2] >= 0: l_vertices2elements[g2l[v2]].append(iElt)
            if g2l[v3] >= 0: l_vertices2elements[g2l[v3]].append(iElt)

        begin_vertex2elements = np.zeros((nb_vertices+1), dtype=np.int)
        for iVert in range(nb_vertices):
            begin_vertex2elements[iVert+1] = begin_vertex2elements[iVert] + len(l_vertices2elements[iVert])
        vertex2elements = np.empty(begin_vertex2elements[-1], dtype=np.int64)
        pt_v2e : int = 0
        for elts in l_vertices2elements:
            vertex2elements[pt_v2e:pt_v2e+len(elts)] = np.array(elts)
            pt_v2e += len(elts)
        return (begin_vertex2elements, vertex2elements)

def read( filename : str ):
    """
    Usage : m, cl = read( filename )
    Lit un maillage à partir d'un fichier formaté donné par filename.
    Le fichier doit être au format gmsh. La fonction retourne un maillage
    de type Mesh. Pour chaque sommet, on renvoit également le type de sommet
    ( 0 c'est un noeud "libre", sinon c'est une condition limite de Dirichlet )
    """
    with open(filename, 'r') as fich:
        data = fich.readlines()
    regex_values = r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    values = re.findall(regex_values, data[1])
    version : float = float(values[0])
    file_type: int  = int(values[1])
    data_size: int  = int(values[2])

    if file_type != 0 or data_size != 8:
        raise IOError("Wrong file type")
    iLine = 2

    vertices = None
    elt2verts= None
    vert_type= None
    while iLine < len(data):
        if data[iLine].find("Nodes") >= 0:
            iLine += 1
            values = re.findall(regex_values, data[iLine])
            iLine += 1
            nbVertices = int(values[0])
            vertices = np.empty((nbVertices,2), dtype=np.double )
            vert_type= np.zeros(nbVertices, dtype=np.int64)
            for iVert in range(nbVertices):
                values = re.findall(regex_values, data[iLine])
                iLine += 1
                ind_node = int(values[0])-1            
                vertices[ind_node,:] = [float(values[1]), float(values[2])]
            iLine += 1 # Pour sauter le $EndNodes
        elif data[iLine].find("Elements") >= 0:
            iLine += 1
            values = re.findall(regex_values, data[iLine])
            iLine += 1
            nbElts  = int(values[0])
            elt2verts = []
            for iElt in range(nbElts):
                values = re.findall(regex_values, data[iLine])
                iLine += 1
                num_elt = int(values[0])-1
                tags    = np.array( [ int(values[ivalue]) for ivalue in range(3,3+int(values[2])) ], dtype=np.int64)
                begInds = 3+int(values[2])
                if int(values[1]) == 1: # Si le type d'élément est de type ligne physique, c'est une C.L
                    nd1 = int(values[begInds])-1
                    nd2 = int(values[begInds+1])-1
                    vert_type[nd1] = tags[-1]
                    vert_type[nd2] = tags[-1]
                elif int(values[1])==2: # Si c'est un triangle :
                    elt2verts.append(np.array([int(values[begInds  ])-1, 
                                               int(values[begInds+1])-1, 
                                               int(values[begInds+2])-1], dtype=np.int64))
            iLine += 1
        else:
            iLine += 1
    element2vertices = np.array(elt2verts, dtype=np.int64)
    return Mesh(vertices, element2vertices), vert_type

if __name__ == '__main__':
    m, cl = read("CarreMedium.msh")
    vert2elts = m.comp_vertices_to_elements()

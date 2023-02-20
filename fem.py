# Gestion d'une matrice creuse FEM
import numpy as np

def compute_skeleton_sparse_matrix( element2dofs, dof2elts ):
    """
    Usage : compute_skeleton_sparse_matrix( elt2dofs, (pt_dof2elts, dof2elts ) )

    Calcul le graphe de la matrice creuse issue des éléments finis.
    
    element2dofs: Tableau donnant pour chaque élément les degrés de liberté associés (shape (nbElt,nbDofPerElt)).
    pt_dof2elts : pointeurs dans le tableau dof2elts donnant pour chaque degré de
                  liberté les éléments associés.
    dof2elts    : tableau donnant pour chaque degré de liberté les éléments associés.
    """
    pt_dof2elts = dof2elts[0]
    ar_dof2elts = dof2elts[1]
    nb_dofs : int = pt_dof2elts.shape[0]-1

    begin_rows = np.empty( nb_dofs+1, dtype=np.int64)
    begin_rows[0] = 0
    for iDof in range(nb_dofs):
        lst_neighbours = []
        for pt_iElt in range(pt_dof2elts[iDof], pt_dof2elts[iDof+1]):
            iElt : int = ar_dof2elts[pt_iElt]
            lst_neighbours.extend(element2dofs[iElt,:])
        lst_neighbours = sorted(set(lst_neighbours))
        begin_rows[iDof+1] = begin_rows[iDof] + len(lst_neighbours)
    nb_non_zeros = begin_rows[-1]
    indice_columns = np.empty(nb_non_zeros, dtype=np.int64)

    for iDof in range(nb_dofs):
        lst_neighbours = []
        for pt_iElt in range(pt_dof2elts[iDof], pt_dof2elts[iDof+1]):
            iElt : int = ar_dof2elts[pt_iElt]
            lst_neighbours.extend(element2dofs[iElt,:])
        lst_neighbours = sorted(set(lst_neighbours))
        indice_columns[begin_rows[iDof]:begin_rows[iDof]+len(lst_neighbours)] = np.array(lst_neighbours)

    return begin_rows, indice_columns
# ------------------------------------------------------------------------------------------------------------------------
def add_elementary_matrix_to_csr_matrix( sparse_matrix, element_matrix, masks = None ):
    """
    Usage : add_elemMat_csrMatrix( (begRows, indCols, coefs), (indRows, indCols, elemMat), [(rowMask, colMask)] )
    
    Rajoute la matrice élémentaire définie par le tuple (indRows, indCols,elemMat)
    à la matrice creuse stockée CSR définie par le tuple (begRows, indCols, coefs).
    """
    sm_begin_rows     = sparse_matrix[0]
    sm_indice_columns = sparse_matrix[1]
    sm_coefficients   = sparse_matrix[2]
    em_indice_rows    = element_matrix[0]
    em_indice_columns = element_matrix[1]
    em_coefficients   = element_matrix[2]

    row_mask = masks[0] if masks is not None else None 
    col_mask = masks[1] if masks is not None else None

    nb_rows_mat_elem : int = em_indice_rows.shape[0]
    nb_cols_mat_elem : int = em_indice_columns.shape[0]

    if row_mask is None:
        for iRow in range(nb_rows_mat_elem):
            indice_row : int = em_indice_rows[iRow]
            for jCol in range(nb_cols_mat_elem):
                indice_col : int = em_indice_columns[jCol]
                pt_col = np.where(sm_indice_columns[sm_begin_rows[indice_row]:sm_begin_rows[indice_row+1]]==indice_col)[0]
                if pt_col.shape[0] > 0:
                    sm_coefficients[pt_col[0]+sm_begin_rows[indice_row]] += em_coefficients[iRow, jCol]
    else:
        for iRow in range(nb_rows_mat_elem):
            indice_row : int = em_indice_rows[iRow]
            if row_mask[indice_row]:
                for jCol in range(nb_cols_mat_elem):
                    indice_col : int = em_indice_columns[jCol]
                    pt_col = np.where(sm_indice_columns[sm_begin_rows[indice_row]:sm_begin_rows[indice_row+1]]==indice_col)[0]
                    if pt_col.shape[0] > 0:
                        sm_coefficients[pt_col[0]+sm_begin_rows[indice_row]] += em_coefficients[iRow,jCol]

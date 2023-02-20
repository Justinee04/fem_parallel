# Calcul la matrice élémentaire du laplacien sur un triangle.
import numpy as np

def compute_elementary_matrix( p1, p2, p3 ) ->np.array :
    p1p3 = (p3[0] - p1[0], p3[1] - p1[1])
    p1p2 = (p2[0] - p1[0], p2[1] - p1[1])
    p2p3 = (p3[0] - p2[0], p3[1] - p2[1])

    p1p3_dot_p1p3 = p1p3[0]*p1p3[0] + p1p3[1]*p1p3[1]
    p1p2_dot_p1p3 = p1p2[0]*p1p3[0] + p1p2[1]*p1p3[1]
    p2p3_dot_p2p3 = p2p3[0]*p2p3[0] + p2p3[1]*p2p3[1]

    A12 = 0.5*( p1p2_dot_p1p3 - p1p3_dot_p1p3 )
    A13 = 0.5*( p1p2_dot_p1p3 - p2p3_dot_p2p3 )
    A23 = -0.5*p1p2_dot_p1p3

    return np.array([ [ 0.5*(p1p3_dot_p1p3 - 2.*p1p2_dot_p1p3 + p2p3_dot_p2p3), A12, A13 ],
                      [ A12, 0.5*p1p3_dot_p1p3, A23],
                      [ A13, A23, 0.5*p2p3_dot_p2p3]])

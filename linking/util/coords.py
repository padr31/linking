import numpy as np

'''
The following functions use the representation where v1, v2, v3
are three consecutively added BFS edges.
'''
def calc_angle(v2, v3):
    """
        Calculate angle between two lines, atan2 is used to avoid
        the ill cosine behaviour around extrema.
    """
    sin_theta = np.linalg.norm(np.cross(v2, v3))
    cos_theta = np.dot(v2, v3)
    return np.arctan2(sin_theta, cos_theta)

def calc_dihedral(v1, v2, v3):
    """
       Calculate dihedral angle between three edges.
    """
    # Normal vector of plane containing v1,v2
    n1 = np.cross(v1, v2)
    n1 = n1 / np.linalg.norm(n1)

    # Normal vector of plane containing v2,v3
    n2 = np.cross(v2, v3)
    n2 = n2 / np.linalg.norm(n2)

    # un1, ub2, and um1 form orthonormal frame
    uv2 = v2 / np.linalg.norm(v2)
    m1 = np.cross(n1, uv2)
    m1 = m1 / np.linalg.norm(m1)

    # dot(ub2, n2) is always zero
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    dihedral = np.arctan2(y, x)
    return dihedral

def calc_position(v1, v2, p3, dst, ang, dih):
    """Calculate position x of another atom based on
       internal coordinates between v1, v2, (p3,x)
       using distance, angle, and dihedral angle.
    """
    # Normal vector of plane containing v1,v2
    n1 = np.cross(v1, v2)
    if np.linalg.norm(n1) == 0:
        print('null')
    n1 = n1 / np.linalg.norm(n1)

    # un1, ub2, and um1 form orthonormal frame
    uv2 = v2 / np.linalg.norm(v2)
    m1 = np.cross(n1, uv2)
    m1 = m1 / np.linalg.norm(m1)

    n2 = np.cos(dih)*n1 + np.sin(dih)*m1
    if np.linalg.norm(n2) == 0.0:
        print('null')
    n2 = n2 / np.linalg.norm(n2)

    nn2 = np.cross(n2, uv2)
    nn2 = nn2 / np.linalg.norm(nn2)
    v3 = np.cos(ang)*uv2 + np.sin(ang)*nn2
    v3 = v3 / np.linalg.norm(v3)

    position = p3 + dst * v3

    return position

'''
The following functions use the representation v1 = p2-p1, v2 = p3-p2, v3 = p4-p3
'''
def calc_angle_p(p2, p3, p4):
    return calc_angle(p3-p2, p4-p3)

def calc_dihedral_p(p1, p2, p3, p4):
    return calc_dihedral(p2-p1, p3-p2, p4-p3)

def calc_position_p(p1, p2, p3, dst, ang, dih):
    return calc_position(p2-p1, p3-p2, p3, dst, ang, dih)
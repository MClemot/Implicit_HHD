# -*- coding: utf-8 -*-

import numpy as np
from itertools import combinations
from pygel3d import hmesh
from scipy.spatial import KDTree

# =============================================================================
# SDFs
# =============================================================================

def phi_sphere(p):
    return np.linalg.norm(p, axis=1) - .5

def phi_torus(p):
    l = np.linalg.norm(p[:,[0,2]], axis=1) - .65
    return np.sqrt(l**2+p[:,1]**2) - .25

def phi_point_cloud(p, pc, nc):
    KDT = KDTree(pc)
    NNv, NNi = KDT.query(p)
    s = (np.sum(nc[NNi]*(p-pc[NNi]), axis=1) > 0)*2 - 1
    return NNv * s

def phi_from_mesh(s):
    N = 1000000
    pc, nc = sample_mesh(s, N)
    return lambda p:phi_point_cloud(p, pc, nc)

# =============================================================================
# level set visualization
# =============================================================================

def same_sign(tet, f, l):
    v = f[tet] - l
    if np.all(v>0) or np.all(v<0):
        return True
    else:
        return False

def not_too_far(tet, f, eps):
    v = np.abs(f[tet])
    if np.all(v<=eps):
        return True
    else:
        return False
    
def intersection(pts, tet, f, l):
    ipts, iedg, coord = [],[],[]
    for e in combinations(tet, 2):
        v0,v1 = f[e[0]]-l, f[e[1]]-l
        if v0*v1 < 0:
            t = -v0 / (v1 - v0)
            p = (1-t)*pts[e[0]] + t*pts[e[1]]
            ipts.append(p)
            iedg.append(e)
            coord.append(t)
    return ipts, iedg, coord


# =============================================================================
# sampling on mesh
# =============================================================================

def sample_mesh(s, N):
    m = hmesh.load(s)
    sample = []
    normals = []
    
    area = 0
    for t in m.faces():
        area += m.area(t)
    
    for t in m.faces():
        n = m.face_normal(t)
        ver = []
        for v in m.circulate_face(t, mode='v'):
            ver.append(v)
        r = m.area(t)/area * N
        if r<1:
            if np.random.rand() > r:
                continue
        num = int(np.ceil(r))
        r1 = np.random.rand(num)
        r2 = np.random.rand(num)
        loc = (np.ones_like(r1)-np.sqrt(r1))[:,None] * m.positions()[ver[0]][None,:]
        loc += (np.sqrt(r1)*(np.ones_like(r1)-r2))[:,None] * m.positions()[ver[1]][None,:]
        loc += (np.sqrt(r1)*r2)[:,None] * m.positions()[ver[2]][None,:]
        for p in loc:
            sample.append(p)
            normals.append(n)
    
    sample = np.array(sample) / (1.1*np.max(np.abs(sample)))

    return sample, np.array(normals)
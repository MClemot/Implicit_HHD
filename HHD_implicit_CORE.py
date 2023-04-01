# -*- coding: utf-8 -*-

import gudhi
import igl
import numpy as np
import scipy.sparse as sparse
import scipy.spatial.transform as transform
import tetgen

from tools import area, volume, barycentric
from level_set import same_sign, intersection

face_indices = [[0,1,2],[0,1,3],[1,2,3],[2,0,3]]
edges_of_tri = [[0,1],[1,2],[2,0]]
edges_of_quad = [[0,1],[1,3],[3,2],[2,0],[1,2]]
faces_of_tet = [[0,1,2],[1,2,3],[2,3,0],[3,0,1]]


def generate_tetmesh(p):
    V,_,_,F,_,_ = igl.read_obj("Objects/cube2.obj")
    V, T = tetgen.TetGen(V, F).tetrahedralize(switches="Qpq1.414a{}".format(p))
    return V, T


def keep_intersecting_tets(V, T, phi, l):
    tmp_T = []
    tmp_V = dict()
    for n,tet in enumerate(T):
        if not same_sign(tet, phi, l):
            for v in tet:
                if not v in tmp_V:
                    tmp_V[v] = len(tmp_V)
            tmp_T.append(tet)
    T = np.array(tmp_T)
    V = V[list(tmp_V)]
    phi = phi[list(tmp_V)]
    for i in range(T.shape[0]):
        for j in range(4):
            T[i,j] = tmp_V[T[i,j]]
            
    return V, T, phi


def check_topology(T, b, g):
    st = gudhi.SimplexTree()
    st.insert_batch(T.T, np.zeros(T.shape[0]))

    n_s = [0,0,0,0]
    for s in st.get_simplices():
        n_s[len(s[0])-1] += 1
    print("Number of simplices:", n_s)

    st.compute_persistence()
    betti = st.betti_numbers()
    print("Betti numbers:", betti)
    if b != None:
        assert(betti == b)
    euler_char = n_s[0]-n_s[1]+n_s[2]-n_s[3]
    print("Euler characteristic:", euler_char)
    if g != None:
        assert(euler_char == 2-2*g)
        
        
def compute_edges_and_faces(T):
    _, tet_tet_adj = igl.tet_tet_adjacency(T)
    
    F = []
    F_index = dict()
    Fi = []
    Fi_index = dict()
    F_all_to_inner = []
    for n,tet in enumerate(T):
        for k in range(4):
            tri = tet[face_indices[k]]
            if frozenset(tri) not in F_index:
                F_index[frozenset(tri)] = len(F)
                F.append(list(tri))
            if tet_tet_adj[n,k] != -1:
                tri = tet[face_indices[k]]
                if frozenset(tri) not in Fi_index:
                    Fi_index[frozenset(tri)] = len(Fi)
                    Fi.append(list(tri))
                    F_all_to_inner.append((len(Fi)-1, len(F)-1))
    F = np.array(F)
    Fi = np.array(Fi)
    E = igl.edges(Fi)
    
    tri_tet_adj = [[] for _ in range(Fi.shape[0])]
    for n,tet in enumerate(T):
        for k in range(4):
            if tet_tet_adj[n,k] != -1:
                tri = tet[face_indices[k]]
                tri_tet_adj[Fi_index[frozenset(tri)]].append(n)
                
    return E, F, F_index, Fi, Fi_index, F_all_to_inner, tri_tet_adj


def compute_gradient_operator(V, T, phi):
    n_V = V.shape[0]
    n_T = T.shape[0]
    
    G_V = sparse.lil_matrix((3*n_T, n_V))
    A_phi = sparse.lil_matrix((n_T, n_V))
    M_X = np.zeros((3*n_T))

    for n,tet in enumerate(T):
        tet_pts = V[tet]
        vol = volume(tet_pts)
        
        p0,p1,p2,p3 = tet_pts[0], tet_pts[1], tet_pts[2], tet_pts[3]
        
        G_V[3*n:3*n+3,tet[0]] = np.cross(p3-p1,p2-p1) / (6*vol)
        G_V[3*n:3*n+3,tet[1]] = np.cross(p3-p2,p0-p2) / (6*vol)
        G_V[3*n:3*n+3,tet[2]] = np.cross(p3-p0,p1-p0) / (6*vol)
        G_V[3*n:3*n+3,tet[3]] = np.cross(p1-p0,p2-p0) / (6*vol)
        
        M_X[3*n:3*n+3] = np.abs(vol)

    grad_phi = G_V.dot(phi)
    G_V_phi = G_V.copy()

    for n,tet in enumerate(T):
        nt = grad_phi[3*n:3*n+3]
        nt /= np.linalg.norm(nt)
        
        for i in range(4):
            current = G_V_phi[3*n:3*n+3,tet[i]].toarray().flatten()
            G_V_phi[3*n:3*n+3,tet[i]] = current - np.dot(current, nt) * nt
        
        for i in range(4):
            A_phi[n,tet[i]] = np.dot(G_V[3*n:3*n+3,tet[i]].toarray().flatten(), nt)    
        
    G_V = G_V.tocsc()
    G_V_phi = G_V_phi.tocsc()
    
    return G_V, G_V_phi, M_X, grad_phi, A_phi


def generate_tangent_vector_field(V, T, grad_phi):
    n_T = T.shape[0]
    v_field = np.zeros((3*n_T))
    
    for n,tet in enumerate(T):
        nt = grad_phi[3*n:3*n+3]
        p = np.mean(V[tet], axis=0)
        v_field[3*n:3*n+3] = np.array([np.sin(1*p[0]+2*p[1]), np.cos(2*p[1]), 2*np.sin(p[0])*np.cos(3*p[2])])
        v_field[3*n:3*n+3] -= np.dot(v_field[3*n:3*n+3], nt) * nt
        
    return v_field


def compute_curl_operator(V, Fi, T, grad_phi, tri_tet_adj):
    n_Fi = Fi.shape[0]
    n_T = T.shape[0]
    C = sparse.lil_matrix((n_Fi,3*n_T))

    for n,face in enumerate(Fi):
        neighbors = tri_tet_adj[n]
        assert(len(neighbors) == 2)
        nt1,nt2 = neighbors[0], neighbors[1]
        
        a = area(V[face])
        gphi1 = grad_phi[3*nt1:3*nt1+3]
        gphi2 = grad_phi[3*nt2:3*nt2+3]
        vec = np.cross(gphi1, gphi2)
        vec /= np.linalg.norm(vec)
        
        C[n,3*nt1:3*nt1+3] = a * vec
        C[n,3*nt2:3*nt2+3] = -a * vec

    C = C.tocsc()
    return C


def compute_level_set(V, T, phi, l, v_field):
    i_pts = []
    i_tri = []
    i_input = []
    edge_to_ptidx = dict()
    Bf_construction = []

    i_edges_set = dict()
    def add_i_edge(e, tet):
        if frozenset(e) not in i_edges_set:
            i_edges_set[frozenset(e)] = [tet]
        else:
            i_edges_set[frozenset(e)].append(tet)

    for n,tet in enumerate(T):
        p, e, coords = intersection(V, tet, phi, l)
        ptsidx = []
        for i in range(len(p)):
            if frozenset(e[i]) in edge_to_ptidx:
                ptsidx.append(edge_to_ptidx[frozenset(e[i])])
            else:
                edge_to_ptidx[frozenset(e[i])] = len(i_pts)
                ptsidx.append(len(i_pts))
                i_pts.append(p[i])
                Bf_construction.append((len(i_pts)-1, e[i], coords[i]))
        
        ptsidx = np.array(ptsidx)
        loc_input = v_field[3*n:3*n+3]
        
        if len(p) == 3:
            i_tri.append(ptsidx)
            i_input.append(loc_input)
            for e in edges_of_tri:
                add_i_edge(ptsidx[e], tet)
                
        elif len(p) == 4:
            i_tri.append(ptsidx[:3])
            i_input.append(loc_input)
            i_tri.append(ptsidx[1:])
            i_input.append(loc_input)
            for e in edges_of_quad:
                add_i_edge(ptsidx[e], tet)

    i_pts = np.array(i_pts)
    i_tri = np.array(i_tri)
    i_input = np.array(i_input)
    
    return i_pts, i_tri, i_input, i_edges_set, Bf_construction


def compute_Bf_interpolator(i_pts, V, Bf_construction):
    B_f = sparse.lil_matrix((i_pts.shape[0], V.shape[0]))
    for p,e,coord in Bf_construction:
        B_f[p, e[0]] = 1-coord
        B_f[p, e[1]] = coord
    B_f = B_f.tocsc()
    
    return B_f


def compute_Bg_interpolator(i_pts, i_edges_set, V, F, F_index):
    B_g = sparse.lil_matrix((len(i_edges_set), F.shape[0]))
    i_edges = []
    for i,e_ in enumerate(i_edges_set):
        e = list(e_)
        i_edges.append(e)
        mid_edge = i_pts[e].mean(axis=0)
        tets = i_edges_set[e_]
        
        for tet in tets:
            faces_indices = []
            mid_faces = []
            for f in faces_of_tet:
                faces_indices.append(F_index[frozenset(tet[f])])
                mid_faces.append(V[tet[f]].mean(axis=0))
            mid_faces = np.array(mid_faces)
            B_g[i, faces_indices] += barycentric(mid_edge, mid_faces) / len(tets)

    i_edges = np.array(i_edges)
    B_g = B_g.tocsc()
    
    return B_g, i_edges


def compute_gradientF_operator(V, F, F_index, T):
    n_F = F.shape[0]
    n_T = T.shape[0]
    G_F = sparse.lil_matrix((3*n_T, n_F))
    for n,tet in enumerate(T):
        dual_tet = []
        dual_tet_pts = []
        for f in faces_of_tet:
            dual_tet.append(F_index[frozenset(tet[f])])
            dual_tet_pts.append(V[tet[f]].mean(axis=0))
        dual_tet_pts = np.array(dual_tet_pts)
        vol = volume(dual_tet_pts)
        p0,p1,p2,p3 = dual_tet_pts[0], dual_tet_pts[1], dual_tet_pts[2], dual_tet_pts[3]
        G_F[3*n:3*n+3,dual_tet[0]] = np.cross(p3-p1,p2-p1) / (6*vol)
        G_F[3*n:3*n+3,dual_tet[1]] = np.cross(p3-p2,p0-p2) / (6*vol)
        G_F[3*n:3*n+3,dual_tet[2]] = np.cross(p3-p0,p1-p0) / (6*vol)
        G_F[3*n:3*n+3,dual_tet[3]] = np.cross(p1-p0,p2-p0) / (6*vol)
    
    G_F = G_F.tocsc()
    return G_F
    

def compute_rotation_operator(T, grad_phi):
    n_T = T.shape[0]
    J = sparse.lil_matrix((3*n_T,3*n_T))
    for n,tet in enumerate(T):
        nt = grad_phi[3*n:3*n+3]
        J[3*n:3*n+3,3*n:3*n+3] = transform.Rotation.from_rotvec(nt * np.pi/2).as_matrix()
        
    J = J.tocsc() 
    return J
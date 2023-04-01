# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sparse

def HHD_explicit(V, E, F, v_field):
    
    v, e, f = V.shape[0], E.shape[0], F.shape[0]
    
    edge_to_pts = dict()
    for face in F:
        i,j,k = face[0], face[1], face[2]
        for edge in [[i,j,k], [j,k,i], [k,i,j]]:
            if frozenset(edge[:2]) not in edge_to_pts:
                edge_to_pts[frozenset(edge[:2])] = []
            edge_to_pts[frozenset(edge[:2])].append(edge[2])

    pts_to_face = dict()
    for nf,face in enumerate(F):
        pts_to_face[frozenset(list(face))] = nf
    
    # =============================================================================
    # gradient
    # =============================================================================

    G_V = sparse.lil_matrix((3*f,v))
    M_X = np.zeros((3*f))

    for nf,face in enumerate(F):
        i,j,k = face[0], face[1], face[2]
        pi,pj,pk = V[i], V[j], V[k]
        n = np.cross(pj-pi, pk-pi)
        n /= np.linalg.norm(n)
        
        a = np.linalg.norm(pi-pj)
        b = np.linalg.norm(pj-pk)
        c = np.linalg.norm(pk-pi)
        p = (a+b+c)/2
        a = np.sqrt(p*(p-a)*(p-b)*(p-c))
        M_X[3*nf:3*nf+3] = a

        G_V[3*nf:3*nf+3,i] = np.cross(n, pk-pj) / (2*a)
        G_V[3*nf:3*nf+3,j] = np.cross(n, pi-pk) / (2*a)
        G_V[3*nf:3*nf+3,k] = np.cross(n, pj-pi) / (2*a)
        
    G_V = G_V.tocsc()
    M_X_inv = sparse.diags(1/M_X, 0, format="csc")
    M_X = sparse.diags(M_X, 0, format="csc")
    D = G_V.transpose() @ M_X

    # =============================================================================
    # curl
    # =============================================================================

    C = sparse.lil_matrix((e,3*f))

    for ne,edge in enumerate(E):
        i,j = edge[0], edge[1]
        vec = V[j]-V[i]
        neighbors = edge_to_pts[frozenset([i,j])]
        assert(len(neighbors) == 2)
        k,l = neighbors[0], neighbors[1]
        nf1 = pts_to_face[frozenset([i,j,k])]
        nf2 = pts_to_face[frozenset([i,j,l])]
        C[ne,3*nf1:3*nf1+3] = vec
        C[ne,3*nf2:3*nf2+3] = -vec

    C = C.tocsc()
    JG_E = M_X_inv @ C.transpose()

    # =============================================================================
    # helmholz hodge
    # =============================================================================

    L_V = G_V.transpose() @ M_X @ G_V
    L_E = C @ JG_E

    hh_f = sparse.linalg.spsolve(L_V, D@v_field)
    grad_f = G_V @ hh_f

    hh_g = sparse.linalg.spsolve(L_E, C@v_field)
    curl_g = JG_E @ hh_g

    hh_h = v_field - grad_f - curl_g
    
    return hh_f, grad_f, hh_g, curl_g, hh_h, G_V, C

# -*- coding: utf-8 -*-

import igl
import numpy as np
import polyscope as ps
import scipy.sparse as sparse

from tools import to_vtk

comparison = False

s = "torus"
V,_,_,F,_,_ = igl.read_obj("Objects/{}.obj".format(s))
if comparison:
    V,_,_,F,_,_ = igl.read_obj("ls.obj")
E = igl.edges(F)
f = F.shape[0]
e = E.shape[0]
v = V.shape[0]
euler_characteristic = igl.euler_characteristic(F)
print("vertices: {}".format(v))
print("edges: {}".format(e))
print("triangles: {}".format(f))
print("Euler characteristic: {} | genus: {}".format(euler_characteristic, (2-euler_characteristic)//2))

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
v_field = np.zeros((3*f))

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
    
    p = 1/3*(pi+pj+pk)
    v_field[3*nf:3*nf+3] = np.array([np.sin(1*p[0]+2*p[1]), np.cos(2*p[1]), 2*np.sin(p[0])*np.cos(3*p[2])])
    v_field[3*nf:3*nf+3] -= np.dot(v_field[3*nf:3*nf+3], n) * n
    
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

# test = C@G_V

if comparison:
    load = np.load("fields.npz")
    i_input, i_grad_f, i_curl_g, i_h = load['arr_0'], load['arr_1'], load['arr_2'], load['arr_3']
    v_field = i_input.flatten()

# =============================================================================
# helmholz hodge
# =============================================================================

L_V = G_V.transpose() @ M_X @ G_V
L_E = C @ JG_E

# print(L_V.shape, np.linalg.matrix_rank(L_V, hermitian=True))
# print(L_E.shape, np.linalg.matrix_rank(L_E, hermitian=True))

hh_f = sparse.linalg.spsolve(L_V, D@v_field)
grad_f = G_V @ hh_f

hh_g = sparse.linalg.spsolve(L_E, C@v_field)
curl_g = JG_E @ hh_g

hh_h = v_field - grad_f - curl_g
print("harmonic part mean absolute value:", np.abs(hh_h).mean())

# test = np.concatenate((G_V, JG_E), axis=1)
# rk = np.linalg.matrix_rank(test)
# print("H nullspace:", 2*f-rk)

# =============================================================================
# implicit to explicit
# =============================================================================

if comparison:
    testcurlfree = C@(i_grad_f.flatten())
    print("G.f", np.abs(testcurlfree).max(), np.abs(testcurlfree).mean())
    testdivfree = D@(i_curl_g.flatten())
    print("JG.g", np.abs(testdivfree).max(), np.abs(testdivfree).mean())
    
    testcurlfree = C@(i_h.flatten())
    print("G.h", np.abs(testcurlfree).max(), np.abs(testcurlfree).mean())
    testdivfree = D@(i_h.flatten())
    print("JG.h", np.abs(testdivfree).max(), np.abs(testdivfree).mean())
    
    diff_grad_f = np.linalg.norm(np.array(np.split(grad_f,f)) - i_grad_f, axis=1) / np.linalg.norm(i_grad_f, axis=1)
    diff_curl_g = np.linalg.norm(np.array(np.split(curl_g,f)) - i_curl_g, axis=1) / np.linalg.norm(i_curl_g, axis=1)
    diff_hh_h = np.linalg.norm(np.array(np.split(hh_h,f)) - i_h, axis=1) / np.linalg.norm(i_h, axis=1)
    print("Mean relative error on G.f", diff_grad_f.mean())
    print("Mean relative error on JG.g", diff_curl_g.mean())
    print("Mean relative error on h", diff_hh_h.mean())

# =============================================================================
# polyscope
# =============================================================================

ps.init()

ps_points = ps.register_point_cloud("Points", V)
ps_mesh = ps.register_surface_mesh("Triangles", V, F)
ps_mesh.set_back_face_policy("identical")
ps_mesh.add_scalar_quantity("Potential", hh_f, defined_on="vertices", enabled=True)

ps_edges = ps.register_curve_network("Edges", V, E)
ps_edges.add_scalar_quantity("Curl Potential", np.abs(hh_g), defined_on='edges')

disp_grad = np.array(np.split(grad_f,f))
disp_curl = np.array(np.split(curl_g,f))
disp_harm = np.array(np.split(hh_h,f))
disp_v = np.array(np.split(v_field,f))

to_vtk("Results/{}".format(s), V, F, [disp_v, disp_grad, disp_curl, disp_harm], ["input", "grad", "curl", "harmo"])
    
ps_mesh.add_vector_quantity("Gradient", disp_grad, defined_on="faces", enabled=True, color=(0,0,1))
ps_mesh.add_vector_quantity("Curl", disp_curl, defined_on="faces", enabled=True, color=(1,0,0))
ps_mesh.add_vector_quantity("Harmonic", disp_harm, defined_on="faces", enabled=True, color=(0,.7,0))
ps_mesh.add_vector_quantity("v field", disp_v, defined_on="faces", enabled=True, color=(0.,0.,0.))
if comparison:
    ps_mesh.add_vector_quantity("Gradient implicit", i_grad_f, defined_on="faces", enabled=True, color=(0,1,1))
    ps_mesh.add_vector_quantity("Curl implicit", i_curl_g, defined_on="faces", enabled=True, color=(1,.75,.75))

ps.show()
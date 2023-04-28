# -*- coding: utf-8 -*-

import igl
import numpy as np
import polyscope as ps
import scipy.sparse as sparse

from tools import area, volume, to_obj, to_vtk, print_elapsed
from level_set import same_sign, intersection, phi_sphere, phi_torus, phi_from_mesh
import HHD_implicit_CORE as Core

s = "Objects/pillowbox1.obj"
s = "torus"

if s == "sphere":
    b = [1,0,1]
    g = 0
    phi = phi_sphere
elif s == "torus":
    b = [1,2,1]
    g = 1
    phi = phi_torus
else:
    phi = phi_from_mesh(s)
    b, g = None, None
    
l = 0

# =============================================================================
# tetrahedral mesh
# =============================================================================
V, T = Core.generate_tetmesh(0.0003)
phi = phi(V)
print("Tetrahedral mesh size:", V.shape[0], T.shape[0])

# =============================================================================
# keeping tets intersecting the level set
# =============================================================================
V, T, phi = Core.keep_intersecting_tets(V, T, phi, l)

# =============================================================================
# topology checks
# =============================================================================
Core.check_topology(T, b, g)

# =============================================================================
# faces
# =============================================================================
E, F, F_index, Fi, Fi_index, _, tri_tet_adj = Core.compute_edges_and_faces(T)

n_V, n_E, n_F, n_Fi, n_T = V.shape[0], E.shape[0], F.shape[0], Fi.shape[0], T.shape[0]
print("Number of faces (inner faces):", "{} ({})".format(n_F, n_Fi))

print_elapsed("Mesh processing")

# =============================================================================
# gradient
# =============================================================================

G_V, G_V_phi, _, M_X, grad_phi = Core.compute_gradient_operator(V, T, phi)

M_X_inv = sparse.diags(1/M_X, 0, format="csc")
M_X = sparse.diags(M_X, 0, format="csc")

v_field = Core.generate_tangent_vector_field(V, T, grad_phi)

D = G_V.transpose() @ M_X

print_elapsed("G_V computation")

# =============================================================================
# curl
# =============================================================================

C = Core.compute_curl_operator(V, Fi, T, grad_phi, tri_tet_adj)
JG_F = M_X_inv @ C.transpose()

print_elapsed("C computation  ")

print("C.Gv test:", np.abs(C@G_V).max())
print("Gv.JGf test:", np.abs(G_V.T@M_X@JG_F).max())

# =============================================================================
# helmholz hodge
# =============================================================================

L_V = G_V.transpose() @ M_X @ G_V
L_E = C @ JG_F

# print(L_V.shape, np.linalg.matrix_rank(L_V.toarray(), hermitian=True))
# print(L_E.shape, np.linalg.matrix_rank(L_E.toarray(), hermitian=True))

# svd = sparse.linalg.svds(L_V, k=min(L_V.shape)-1, tol=1e-10)
# print(len(svd[1]))

hh_f = sparse.linalg.spsolve(L_V, D@v_field)
grad_f = G_V @ hh_f

hh_g = sparse.linalg.spsolve(L_E, C@v_field)
curl_g = JG_F @ hh_g

hh_h = v_field - grad_f - curl_g
print("harmonic part abs mean", np.abs(hh_h).mean())

print_elapsed("HH decomposition")

# print(D.shape, np.linalg.matrix_rank(D))
# print(C.shape, np.linalg.matrix_rank(C))

# =============================================================================
# level set
# =============================================================================

i_pts = []
i_tri = []
i_input = []
i_grad_f = []
i_curl_g = []
i_h = []
edge_to_ptidx = dict()

for n,tet in enumerate(T):
    p,e,_ = intersection(V, tet, phi, l)
    ptsidx = []
    for i in range(len(p)):
        if frozenset(e[i]) in edge_to_ptidx:
            ptsidx.append(edge_to_ptidx[frozenset(e[i])])
        else:
            edge_to_ptidx[frozenset(e[i])] = len(i_pts)
            ptsidx.append(len(i_pts))
            i_pts.append(p[i])
            
    loc_input = v_field[3*n:3*n+3]
    loc_grad_f = grad_f[3*n:3*n+3]
    loc_curl_g = curl_g[3*n:3*n+3]
    loc_h = hh_h[3*n:3*n+3]
    if len(p) == 3:
        i_tri.append(ptsidx)
        i_input.append(loc_input)
        i_grad_f.append(loc_grad_f)
        i_curl_g.append(loc_curl_g)
        i_h.append(loc_h)
    elif len(p) == 4:
        i_tri.append(ptsidx[:3])
        i_input.append(loc_input)
        i_grad_f.append(loc_grad_f)
        i_curl_g.append(loc_curl_g)
        i_h.append(loc_h)
        i_tri.append(ptsidx[1:])
        i_input.append(loc_input)
        i_grad_f.append(loc_grad_f)
        i_curl_g.append(loc_curl_g)
        i_h.append(loc_h)

to_obj(i_pts, "ls.obj", tri=i_tri)

i_pts = np.array(i_pts)
i_tri = np.array(i_tri)
i_input = np.array(i_input)
i_grad_f = np.array(i_grad_f)
i_curl_g = np.array(i_curl_g)
i_h = np.array(i_h)

np.savez("fields.npz", i_input, i_grad_f, i_curl_g, i_h)

print_elapsed("Polyscope")

# =============================================================================
# polyscope
# =============================================================================

if True:
    ps.init()
    
    ps_points = ps.register_point_cloud("Points", i_pts, enabled=False)
    ps_mesh = ps.register_surface_mesh("Triangles", i_pts, i_tri)
    ps_mesh.set_back_face_policy("identical")
    ps_mesh.add_vector_quantity("v field", i_input, defined_on="faces", color=(0,0,0))
    ps_mesh.add_vector_quantity("Gradient", i_grad_f, defined_on="faces", color=(0,0,1))
    ps_mesh.add_vector_quantity("Curl", i_curl_g, defined_on="faces", color=(1,0,0))
    ps_mesh.add_vector_quantity("Harmonic", i_h, defined_on="faces", color=(0,.7,0))
    
    # ps_vol = ps.register_volume_mesh("Tetrahedral mesh", V, tets=T)
    # ps_vol.add_vector_quantity("Gradient", disp_grad, defined_on="cells", enabled=True)
    
    ps.show()
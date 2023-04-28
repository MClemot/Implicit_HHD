# -*- coding: utf-8 -*-

import igl
import numpy as np
import polyscope as ps
import scipy.sparse as sparse
import scipy.linalg as linalg

from tools import print_elapsed
from level_set import phi_sphere, phi_torus, phi_from_mesh
from HHD_explicit_CORE import HHD_explicit
import HHD_implicit_CORE as Core

s = "Objects/pillowbox1.obj"
s = "sphere"

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
V, T = Core.generate_tetmesh(0.001)
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
E, F, F_index, Fi, Fi_index, F_all_to_inner, tri_tet_adj = Core.compute_edges_and_faces(T)

n_V, n_E, n_F, n_Fi, n_T = V.shape[0], E.shape[0], F.shape[0], Fi.shape[0], T.shape[0]
print("Number of faces (inner faces):", "{} ({})".format(n_F, n_Fi))

K = sparse.lil_matrix((n_Fi, n_F))
for x in F_all_to_inner:
    K[x[0], x[1]] = 1
K = K.tocsc()

print_elapsed("Mesh processing")

# =============================================================================
# gradient
# G_V     : (3*n_T, n_V)
# A_phi   : (n_T, n_V)
# M_X     : (3*n_T)
# v_field : (3*n_T)
# =============================================================================
G_V, G_V_phi, G_C, M_X, grad_phi = Core.compute_gradient_operator(V, T, phi)

M_X_inv = sparse.diags(1/M_X, 0, format="csc")
M_X_sqrt = sparse.diags(np.sqrt(M_X/np.sum(M_X)), 0, format="csc")
M_X_inv_sqrt = sparse.diags(1/np.sqrt(M_X/np.sum(M_X)), 0, format="csc")
M_X = sparse.diags(M_X, 0, format="csc")

v_field = Core.generate_tangent_vector_field(V, T, grad_phi)

print_elapsed("G_V computation")

# =============================================================================
# curl
# C : (n_Fi,3*n_T)
# =============================================================================
C, C3 = Core.compute_curl_operator(V, Fi, T, grad_phi, tri_tet_adj)

print_elapsed("C computation")

# =============================================================================
# level set and explicit
# =============================================================================
i_pts, i_tri, i_input, i_edges_set, Bf_construction = Core.compute_level_set(V, T, phi, grad_phi, l, v_field)

B_f = Core.compute_Bf_interpolator(i_pts, V, Bf_construction)
B_g, i_edges = Core.compute_Bg_interpolator_from_corners(i_pts, i_edges_set, V, T)

n_E0 = len(i_edges_set)

print_elapsed("Level-set")

explicit = True
if explicit:
    hh_f0_, grad_f0_, hh_g0_, curl_g0_, hh_h0, G0_V, C0 = HHD_explicit(i_pts, i_edges, i_tri, i_input.flatten())
    n_F0 = i_tri.shape[0]
    
    print_elapsed("Explicit HHD")

# =============================================================================
# exact resolution
# =============================================================================
l = 1

A_f = sparse.vstack((M_X_sqrt@(G_V_phi-G_V),
                     np.sqrt(l)*B_f))
P_f = l * linalg.solve((A_f.T@A_f).toarray(), B_f.T.toarray())

hh_f0 = sparse.linalg.lsqr(M_X_sqrt @ G_V @ P_f, M_X_sqrt @ v_field) [0]
hh_f = P_f @ hh_f0

print(np.abs(B_f @ hh_f - hh_f0).mean())
print(np.abs(G_V_phi @ hh_f - G_V @ hh_f).mean())
# hh_f0 -= hh_f0.mean()
# hh_f -= hh_f.mean()
grad_f = G_V_phi @ hh_f

print_elapsed("Exact Poisson")

# =============================================================================
# co-exact resolution
# G_C : (3*n_T,4*n_T)
# J   : (3*n_T,3*n_T)
# =============================================================================

J = Core.compute_rotation_operator(T, grad_phi)
K_c = Core.compute_continuity_operator(V, T)
K = sparse.hstack((K_c, sparse.csc_matrix((K_c.shape[0],n_Fi))))
L_K = K.T @ K

method = ["full LS", "constrained LS"] [1]

if method == "constrained LS":
    BC = sparse.csc_matrix((2,4*n_T+n_Fi))
    BC[0,0] = 1.
    BC[1,4*n_T] = 1.
    
    A_g = sparse.vstack((BC,
                         sparse.hstack((J@G_C, -M_X_inv@C.T)),
                         sparse.hstack((B_g  , sparse.csc_matrix((n_E0, n_Fi))))))
    # b_g = np.concatenate((np.zeros(3*n_T+2), hh_g0_))
    
    R = linalg.null_space(A_g.toarray())
    test = R.T @ L_K @ R
    print(test.shape, np.linalg.matrix_rank(test))
    
    Jinv_e = sparse.vstack((sparse.csc_matrix((3*n_T+2, n_E0)),
                            sparse.eye(n_E0, format='csc')))
    tmp = A_g.T @ (linalg.inv((A_g@A_g.T).toarray())) @ Jinv_e
    P_g = tmp - R @ linalg.solve(R.T @ L_K @ R, R.T @ L_K @ tmp)
    
if method == "full LS":
    l = 1
    A_g = sparse.vstack((sparse.hstack((K_c           , sparse.csc_matrix((K_c.shape[0], n_Fi)))),
                         sparse.hstack((J@G_C         , -M_X_inv@C.T)),
                         sparse.hstack((np.sqrt(l)*B_g, sparse.csc_matrix((B_g.shape[0], n_Fi))))))
    
    P_g = l * linalg.solve((A_g.T@A_g).toarray(), sparse.hstack((B_g , sparse.csc_matrix((B_g.shape[0], n_Fi)))).T.toarray())

J_c = sparse.hstack((sparse.eye(4*n_T, format='csc'), sparse.csc_matrix((4*n_T, n_Fi))))
J_f = sparse.hstack((sparse.csc_matrix((n_Fi, 4*n_T)), sparse.eye(n_Fi, format='csc')))

hh_g0 = sparse.linalg.lsqr(M_X_sqrt @ M_X_inv @ C.T @ J_f @ P_g, M_X_sqrt @ v_field) [0]
hh_g = P_g @ hh_g0
hh_g_c = J_c @ hh_g
hh_g_f = J_f @ hh_g

curl_g = M_X_inv @ C.T @ hh_g_f
curl_g2 = J @ G_C @ hh_g_c

print(np.abs(K_c @ hh_g_c).mean())
print(np.abs(B_g @ hh_g_c - hh_g0).mean(), "(should be epsilon)")
print(np.abs(curl_g-curl_g2).mean(), "(should be epsilon)")

print_elapsed("Co-exact Poisson corners")
    

# =============================================================================
# harmonic resolution
# =============================================================================

H_2 = sparse.vstack((P_f.T @ G_V_phi.T @ M_X,
                     (J_f@P_g).T @ C))

H_3 = sparse.vstack((G_V.T @ M_X,
                     C3))

# =============================================================================
# polyscope
# =============================================================================

V_corners = []
for tet in T:
    for i in range(4):
        V_corners.append(V[tet[i]])
V_corners = np.array(V_corners)

if True:
    ps.init()
    
    ps_points = ps.register_point_cloud("Points", i_pts, enabled=False)
    ps_mesh = ps.register_surface_mesh("Triangles", i_pts, i_tri)
    ps_mesh.set_back_face_policy("identical")
    ps_edges = ps.register_curve_network("Edges", i_pts, i_edges, enabled=False)
    
    ps_corners = ps.register_point_cloud("Corners", V_corners)
    ps_corners.add_scalar_quantity("g_c", hh_g_c)
    
    ps_mesh.add_vector_quantity("v field", i_input, defined_on="faces", color=(0,0,0))
    ps_mesh.add_scalar_quantity("Grad potential", hh_f0, defined_on="vertices")
    ps_edges.add_scalar_quantity("Curl potential", hh_g0, defined_on="edges")
    
    if explicit:
        ps_mesh.add_vector_quantity("Gradient", np.array(np.split(grad_f0_,n_F0)), defined_on="faces", color=(0,0,1))
        ps_mesh.add_vector_quantity("Curl", np.array(np.split(curl_g0_,n_F0)), defined_on="faces", color=(1,0,0))
        ps_mesh.add_vector_quantity("Harmonic", np.array(np.split(hh_h0,n_F0)), defined_on="faces", color=(0,1,0))
    
    ps_vol = ps.register_volume_mesh("Tetrahedral mesh", V, tets=T, enabled=True)
    ps_vol.set_transparency(0.5)
    ps_vol.add_vector_quantity("Gradient", np.array(np.split(grad_f,n_T)), defined_on="cells", enabled=False, color=(0,0,1))
    ps_vol.add_vector_quantity("JGradient", np.array(np.split(curl_g,n_T)), defined_on="cells", enabled=True, color=(1,0,0))
    ps_vol.add_vector_quantity("JGradient2", np.array(np.split(curl_g2,n_T)), defined_on="cells", enabled=False)
    
    ps_V = ps.register_point_cloud("V", V, enabled=False)
    ps_V.add_scalar_quantity("Ambient potential", hh_f)
    
    ps_mesh2 = ps.register_surface_mesh("Ambient triangles", V, Fi, enabled=False)
    ps_mesh2.add_scalar_quantity("Ambient curl potential", hh_g_f, defined_on='faces')
    
    ps.show()
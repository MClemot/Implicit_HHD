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
G_V, G_V_phi, M_X, grad_phi, A_phi = Core.compute_gradient_operator(V, T, phi)

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
C = Core.compute_curl_operator(V, Fi, T, grad_phi, tri_tet_adj)

print_elapsed("C computation")

# =============================================================================
# level set
# =============================================================================
i_pts, i_tri, i_input, i_edges_set, Bf_construction = Core.compute_level_set(V, T, phi, l, v_field)

B_f = Core.compute_Bf_interpolator(i_pts, V, Bf_construction)
B_g, i_edges = Core.compute_Bg_interpolator(i_pts, i_edges_set, V, F, F_index)

print_elapsed("Level-set")

# =============================================================================
# exact resolution
# =============================================================================

l = 1

A_f = sparse.vstack((M_X_sqrt@(G_V_phi-G_V),
                     np.sqrt(l)*B_f))
P_f = l * np.linalg.inv((A_f.T@A_f).toarray()) @ B_f.T.toarray()

hh_f0 = sparse.linalg.lsqr(M_X_sqrt @ G_V_phi @ P_f, M_X_sqrt @ v_field,
                            damp=.01, x0=np.zeros((i_pts.shape[0]))) [0]
hh_f = P_f @ hh_f0
# hh_f0 -= hh_f0.mean()
# hh_f -= hh_f.mean()
grad_f = G_V_phi @ hh_f

print_elapsed("Exact Poisson")

# =============================================================================
# co-exact resolution
# G_F : (3*n_T, n_F)
# J   : (3*n_T,3*n_T)
# =============================================================================

G_F = Core.compute_gradientF_operator(V, F, F_index, T)
J = Core.compute_rotation_operator(T, grad_phi)

A_g = sparse.vstack((M_X_sqrt @ (J @ G_F - M_X_inv @ C.T @ K),
                     np.sqrt(l)*B_g))
P_g = l * np.linalg.inv((A_g.T@A_g).toarray()) @ B_g.T.toarray()

hh_g0 = sparse.linalg.lsqr(M_X_inv_sqrt @ C.T @ K @ P_g, M_X_sqrt @ v_field,
                           damp=.01, x0=np.zeros((i_edges.shape[0]))) [0]
hh_g = P_g @ hh_g0
curl_g = M_X_inv @ C.T @ K @ hh_g

print_elapsed("Co-exact Poisson")

# =============================================================================
# extrapoling explicit
# =============================================================================

explicit = False
if explicit:
    hh_f0_, grad_f0_, hh_g0_, curl_g0_, hh_h0, G0_V, C0 = HHD_explicit(i_pts, igl.edges(i_tri), i_tri, i_input.flatten())
    n_F0 = i_tri.shape[0]
        
    grad_f0 = G0_V@hh_f0
    
    print_elapsed("Explicit HHD")
# =============================================================================
# polyscope
# =============================================================================

# extrapol_g = P_g @ hh_g0_
# test_jg = M_X_inv @ C.T @ K @ extrapol_g
# print(B_g @ extrapol_g - hh_g0_)

if True:
    ps.init()
    
    ps_points = ps.register_point_cloud("Points", i_pts, enabled=False)
    ps_mesh = ps.register_surface_mesh("Triangles", i_pts, i_tri)
    ps_mesh.set_back_face_policy("identical")
    ps_edges = ps.register_curve_network("Edges", i_pts, i_edges, enabled=False)
    
    ps_mesh.add_vector_quantity("v field", i_input, defined_on="faces", color=(0,0,0))
    ps_mesh.add_scalar_quantity("Grad potential", hh_f0, defined_on="vertices")
    ps_edges.add_scalar_quantity("Curl potential", hh_g0, defined_on="edges")
    
    if explicit:
        ps_mesh.add_vector_quantity("Gradient", np.array(np.split(grad_f0,n_F0)), defined_on="faces", color=(0,0,1))
        ps_mesh.add_vector_quantity("Curl", np.array(np.split(curl_g0_,n_F0)), defined_on="faces", color=(1,0,0))
        ps_mesh.add_vector_quantity("Harmonic", np.array(np.split(hh_h0,n_F0)), defined_on="faces", color=(0,1,0))
    
    ps_vol = ps.register_volume_mesh("Tetrahedral mesh", V, tets=T, enabled=False)
    ps_vol.set_transparency(0.5)
    ps_vol.add_vector_quantity("Gradient", np.array(np.split(grad_f,n_T)), defined_on="cells", enabled=False)
    ps_vol.add_vector_quantity("JGradient", np.array(np.split(curl_g,n_T)), defined_on="cells", enabled=True)
    
    ps_V = ps.register_point_cloud("V", V, enabled=False)
    ps_V.add_scalar_quantity("Ambient potential", hh_f)
    
    ps_mesh2 = ps.register_surface_mesh("Ambient triangles", V, F)
    ps_mesh2.add_scalar_quantity("Ambient curl potential", hh_g, defined_on='faces')
    
    ps.show()
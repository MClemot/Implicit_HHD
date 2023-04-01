# -*- coding: utf-8 -*-

import igl
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import scipy.sparse as sparse
import scipy.optimize as optim

from tools import area, volume, print_elapsed
from level_set import same_sign, not_too_far, intersection, phi_sphere, phi_torus, phi_from_mesh
import HHD_implicit_CORE as Core

name = "Objects/pillowbox1.obj"
name = "torus"

if name == "sphere":
    b = [1,0,1]
    g = 0
    phi = phi_sphere
elif name == "torus":
    b = [1,2,1]
    g = 1
    phi = phi_torus
else:
    phi = phi_from_mesh(name)
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
E, F, F_index, Fi, Fi_index, _, tri_tet_adj = Core.compute_edges_and_faces(T)

n_V, n_E, n_F, n_Fi, n_T = V.shape[0], E.shape[0], F.shape[0], Fi.shape[0], T.shape[0]
print("Number of faces (inner faces):", "{} ({})".format(n_F, n_Fi))

print_elapsed("Mesh processing")

# =============================================================================
# gradient
# =============================================================================
G_V, G_V_phi, M_X, grad_phi, A_phi = Core.compute_gradient_operator(V, T, phi)

M_X_inv = sparse.diags(1/M_X, 0, format="csc")
M_X_sqrt = sparse.diags(np.sqrt(M_X/np.sum(M_X)), 0, format="csc")
M_X_inv_sqrt = sparse.diags(1/np.sqrt(M_X/np.sum(M_X)), 0, format="csc")
M_X = sparse.diags(M_X, 0, format="csc")

v_field = Core.generate_tangent_vector_field(V, T, grad_phi)

D = G_V_phi.transpose() @ M_X

print_elapsed("G_V computation")

# =============================================================================
# curl
# =============================================================================

C = Core.compute_curl_operator(V, Fi, T, grad_phi, tri_tet_adj)
JG_F = M_X_inv @ C.transpose()

print_elapsed("C computation")

print("C.Gv test:", np.abs(C@G_V).max())
print("C.Gphiv test:", np.abs(C@G_V_phi).max())
print("Gv.JGf test:", np.abs(G_V.T@M_X@JG_F).max())

# =============================================================================
# helmholz hodge
# =============================================================================

# exact and co-exact

hh_f = sparse.linalg.lsqr(M_X_sqrt @ G_V_phi, M_X_sqrt @ v_field) [0]
grad_f = G_V_phi @ hh_f

hh_g = sparse.linalg.lsqr(M_X_inv_sqrt @ C.transpose(), M_X_sqrt @ v_field) [0]
curl_g = JG_F @ hh_g

# hh_h = v_field - grad_f - curl_g

# harmonic

if False:
    A_phi_vec = sparse.lil_matrix((n_T, 3*n_T))
    for n,tet in enumerate(T):
        A_phi_vec[n,3*n:3*n+3] = grad_phi[3*n:3*n+3]
    A_phi_vec = A_phi_vec.tocsc()
    H = sparse.vstack([D, C, A_phi_vec])
    print("constraints shape:", H.shape)
    print("SVD:", sparse.linalg.svds(H, which='SM'))
    constraint = optim.LinearConstraint(H, 0, 0)
    
    def obj(h):
        return np.linalg.norm(h - v_field)**2
    
    hess = 2*sparse.eye(3*n_T)
    hh_h = optim.minimize(obj, np.zeros(3*n_T), method='trust-constr',
                          jac=lambda h:2*(h-v_field), hess=lambda h:hess,
                          constraints=[constraint], options={'verbose': 0}) .x
else:
    hh_h = np.zeros_like(grad_f)

print_elapsed("HH decomposition")

# =============================================================================
# level set
# =============================================================================

def update_levelset(l):
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
            i_tri.append(ptsidx[1:])
            i_input.append(loc_input)
            i_input.append(loc_input)
            i_grad_f.append(loc_grad_f)
            i_grad_f.append(loc_grad_f)
            i_curl_g.append(loc_curl_g)
            i_curl_g.append(loc_curl_g)
            i_h.append(loc_h)
            i_h.append(loc_h)
    
    if len(i_tri) == 0:
        return
    
    i_pts = np.array(i_pts)
    i_tri = np.array(i_tri)
    i_input = np.array(i_input)
    i_grad_f = np.array(i_grad_f)
    i_curl_g = np.array(i_curl_g)
    i_h = np.array(i_h)
    
    # ps_points = ps.register_point_cloud("Points", i_pts, enabled=False)
    ps_mesh = ps.register_surface_mesh("Triangles", i_pts, i_tri)
    ps_mesh.set_back_face_policy("identical")
    ps_mesh.add_vector_quantity("v field", i_input, defined_on="faces", color=(0,0,0))
    ps_mesh.add_vector_quantity("Gradient", i_grad_f, defined_on="faces", color=(0,0,1))
    ps_mesh.add_vector_quantity("Curl", i_curl_g, defined_on="faces", color=(1,0,0))
    ps_mesh.add_vector_quantity("Harmonic", i_h, defined_on="faces", color=(0,.7,0))
    
update_levelset(0.)
print_elapsed("Level-set")

# =============================================================================
# polyscope
# =============================================================================

levelset = 0.
ps.init()

# ps_vol = ps.register_volume_mesh("Tetrahedral mesh", V, tets=T)
# ps_vol.add_vector_quantity("Gradient", np.array(np.split(curl_g,n_T)), defined_on="cells", enabled=True)

def callback():
    global levelset
    changed, levelset = psim.SliderFloat("l", levelset, v_min=-.5, v_max=.5)
    pass
    if changed and levelset:
        update_levelset(levelset)

ps.set_user_callback(callback)
ps.show()
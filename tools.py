# -*- coding: utf-8 -*-

import numpy as np
import pyevtk.hl
from pyevtk.vtk import VtkTriangle
from colorama import Fore, Style, init, deinit
import time

deinit()
init(autoreset=True)

t = time.time()

# =============================================================================
# Debug
# =============================================================================

def print_elapsed(s):
    global t
    print(Style.BRIGHT + Fore.YELLOW + s + ": {:.2f} sec".format(time.time() - t))
    t = time.time()

# =============================================================================
# geometry
# =============================================================================

def area(tri):
    p0,p1,p2 = tri[0], tri[1], tri[2]
    a,b,c = np.linalg.norm(p0-p1), np.linalg.norm(p1-p2), np.linalg.norm(p2-p0)
    p = (a+b+c)/2
    return np.sqrt(p*(p-a)*(p-b)*(p-c))

def volume(tet):
    vec = tet[:3,:] - tet[3,:]
    return -1/6*np.linalg.det(np.array(vec))

def gradient(pts, tet, f):
    tet_pts = pts[tet]
    vol = volume(tet_pts)
    
    p0,p1,p2,p3 = tet_pts[0], tet_pts[1], tet_pts[2], tet_pts[3]

    g0 = np.cross(p3-p1,p2-p1) / (6*vol)
    g1 = np.cross(p3-p2,p0-p2) / (6*vol)
    g2 = np.cross(p3-p0,p1-p0) / (6*vol)
    g3 = np.cross(p1-p0,p2-p0) / (6*vol)
    
    return f[tet[0]]*g0 + f[tet[1]]*g1 + f[tet[2]]*g2 + f[tet[3]]*g3

def barycentric(p, basis):
    M = basis - p
    d = -np.linalg.det(basis[:3,:] - basis[3,:])
    
    return np.array([np.linalg.det(M[[1,2,3],:]),
                     np.linalg.det(M[[0,3,2],:]),
                     np.linalg.det(M[[0,1,3],:]),
                     np.linalg.det(M[[0,2,1],:])]) / d

# =============================================================================
# IO
# =============================================================================

def to_obj(pts, path, tri = None, lines = None):
    f = open(path, "w")
    for p in pts:
        f.write("v {} {} {}\n".format(p[0], p[1], p[2]))
    if tri != None:
        for t in tri:
            l = list(t)
            f.write("f {} {} {}\n".format(l[0]+1, l[1]+1, l[2]+1))
    if lines != None:
        for li in lines:
            l = list(li)
            f.write("l {} {}\n".format(l[0]+1, l[1]+1))
    f.close()
    
def to_csv(data, path):
    f = open(path, 'w')
    for i in range(data.shape[0]):
        f.write("{},{},{}\n".format(data[i,0], data[i,1], data[i,2]))
    f.close()
    
def to_vtk(s, pts, tri, data, dataname):
    x = np.array(pts[:,0], order='C')
    y = np.array(pts[:,1], order='C')
    z = np.array(pts[:,2], order='C')
    
    connectivity  = tri.flatten()
    offsets = np.array([3*i for i in range(len(tri))])
    cell_types = np.array([VtkTriangle.tid for _ in range(len(tri))])
    
    adj_pts_tri = [[] for _ in range(len(pts))]
    for n,t in enumerate(tri):
        for p in t:
            adj_pts_tri[p].append(n)
    
    pointdata = dict()
    for k,d in enumerate(data):
        pdata = np.zeros((len(pts),3))
        for p in range(len(pts)):
            pdata[p] = d[adj_pts_tri[p]].mean(axis=0)
        datax = np.array(pdata[:,0], order='C')
        datay = np.array(pdata[:,1], order='C')
        dataz = np.array(pdata[:,2], order='C')
        pointdata[dataname[k]] = (datax, datay, dataz)
    
    pyevtk.hl.unstructuredGridToVTK(s, x, y, z, connectivity, offsets, cell_types,
                                    pointData=pointdata)
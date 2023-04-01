#include <iostream>
#include <unordered_set>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include <igl/readOBJ.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/edges.h>

using namespace std;
using namespace igl;
using namespace Eigen;

MatrixXd phi_sphere(MatrixXd V) {
    return V.rowwise().norm() - .5*MatrixXd::Ones(V.rows(), V.cols());
}

double phi_torus(Vector3d p) {

    double l = sqrt(p(0)*p(0)+p(2)*p(2)) - .65;
    return sqrt(l*l + p(1)*p(1)) - .25;
}

void same_sign() {

}

int main()
{
    MatrixXd meshV;
    MatrixXi meshE;
    MatrixXi meshF;
    igl::readOBJ("../../Objects/cube2.obj", meshV, meshF);

    MatrixXd V;
    MatrixXi T;
    MatrixXi F;

    igl::copyleft::tetgen::tetrahedralize(meshV, meshF, "pq1.414a0.001", V,T,F);

    MatrixXi E;
    igl::edges(T,E);

    unordered_set<Vector3i, std::less<int>, aligned_allocator<std::pair<const int, Vector3i> > > F_set;
    for(int n_T; n_T < T.rows(); ++n_T) {

    }

    MatrixXd phi = phi_sphere(V);

    polyscope::init();

    polyscope::registerSurfaceMesh("surface", meshV, meshF);
    polyscope::registerSurfaceMesh("volume", V, F);

    polyscope::show();

    return 0;
}

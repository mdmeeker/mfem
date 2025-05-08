//               Generate a NURBS mesh and its LOR version from scratch
//
// Compile with: make nurbs_lor_cartesian
//
// Sample runs:  nurbs_lor_cartesian -d 2 -n 3 -o 3
//
// Description:  This example code generates a NURBS mesh "from scratch"
//               by building up a patch topology mesh and patches. A LOR
//               version of the mesh is then generated using an interpolant
//               defined by interp_rule.
//
//               Interpolation rules (-interp):
//                 - 0: Greville points (default)
//                 - 1: Botella points
//                 - 2: Demko points
//                 - 3: Uniform points

#include "mfem.hpp"
#include <iostream>

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int dim = 2;
   int np = 1;
   int order = 1;
   int interp_rule = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-d", "--dim",
                  "Dimension of the mesh (1, 2, or 3).");
   args.AddOption(&np, "-n", "--num-patches",
                  "Number of patches in the mesh per dimension.");
   args.AddOption(&order, "-o", "--order",
                  "Order of nurbs bases.");
   args.AddOption(&interp_rule, "-interp", "--interpolation-rule",
                  "Interpolation Rule: 0 - Greville, 1 - Botella, 2 - Demko");
   args.Parse();

   // Print & verify options
   args.PrintOptions(cout);
   NURBSInterpolationRule sptype = static_cast<NURBSInterpolationRule>(interp_rule);
   const real_t L = static_cast<real_t>(np); // Length of the mesh in each dimension
   const int NP = pow(np, dim); // Total number of patches in the mesh
   const int pdim = dim + 1; // Projective/homogeneous dimension

   // 2. Create the patch-topology mesh
   Mesh patchTopo;
   if (dim == 1)
   {
      // patchTopo = new Mesh::MakeCartesian1D(NP, L);
      patchTopo = Mesh::MakeCartesian1D(np, L);
   }
   else if (dim == 2)
   {
      patchTopo = Mesh::MakeCartesian2D(np, np, Element::QUADRILATERAL, true, L, L);
   }
   else if (dim == 3)
   {
      patchTopo = Mesh::MakeCartesian3D(np, np, np, Element::HEXAHEDRON, L, L,L);
   }
   else
   {
      MFEM_ABORT("Invalid dimension");
   }

   // 3. Create the reference knotvectors and control points
   //    for each patch (same in all dimensions)
   Array<const KnotVector*> kv_ref(np);
   std::vector<Vector> cpts_ref(np);
   Vector knots;  // knot values
   Vector x; // physical coordinates to interpolate
   int nel; // Number of elements
   int ncp; // Number of control points
   for (int I = 0; I < np; I++)
   {
      nel = I + 1;
      // For a spline basis with C^{-1} continuity at ends and
      // C^{p-1} continuity at interior knots, NCP = order + NEL
      ncp = order + nel;

      cout << endl << "I = " << I << endl;

      // Define knot vectors
      knots.SetSize(nel+1);

      for (int i = 0; i < nel+1; i++)
      {
         knots[i] = (real_t)i / nel;
      }
      kv_ref[I] = new KnotVector(order, knots);

      // Debugging
      cout << "knots :" << endl;
      kv_ref[I]->Print(cout);

      // Coordinates to interpolate
      x.SetSize(ncp);
      for (int i = 0; i < ncp; i++)
      {
         x[i] = (real_t)i / (ncp-1) + I;
      }
      // Find control points that interpolate the coordinates
      cpts_ref[I].SetSize(ncp);
      kv_ref[I]->GetInterpolant(x, NURBSInterpolationRule::Uniform, cpts_ref[I]);

      // Debugging
      cout << "cpts_ref[I] :" << endl;
      cpts_ref[I].Print(cout);
   }


   // // 4. Create the patches
   // Array<NURBSPatch*> patches(NP);
   // int I,J,K; // patch indices
   // int i,j,k; // dof indices
   // int dofidx;
   // Array<int> ncp(3); // number of control points per dim

   // for (int p = 0; p < NP; p++)
   // {
   //    Array<const KnotVector*> kvs(dim);
   //    I = p % np;
   //    J = (p / np) % np;
   //    K = p / (np * np);
   //    int IJK[3] = {I,J,K};

   //    // Collect the knot vectors for this patch
   //    ncp = 1;
   //    for (int d = 0; d < dim; d++)
   //    {
   //       kvs[d] = kv_ref[IJK[d]];
   //       // For a spline basis with C^{-1} continuity at ends and
   //       // C^{p-1} continuity at interior knots, NCP = order + NEL
   //       ncp[d] *= (order + IJK[d]+1);
   //    }

   //    // Define the control points for this patch
   //    // The domain for each patch in physical space is [I, I+1] x [J, J+1] x [K, K+1]
   //    Array<real_t> control_points(pdim * ncp.Prod());
   //    for (int k = 0; k < ncp[2]; k++)
   //    {
   //       for (int j = 0; j < ncp[1]; j++)
   //       {
   //          for (int i = 0; i < ncp[0]; i++)
   //          {
   //             dofidx = i + j*ncp[0] + k*ncp[0]*ncp[1];
   //             int ijk[3] = {i,j,k};
   //             real_t x = I + static_cast<real_t>(i) / (ncp[0]-1);

   //             // Set the control points (+ weight) for the LO mesh
   //             for (int d = 0; d < dim; d++)
   //             {
   //                // control_points[pdim*dofidx + d] = vals[d];
   //                // control_points[pdim*dofidx + d] = IJK[d] + static_cast<real_t>(i) / (ncp[0]-1.0);
   //             }
   //             control_points[pdim*dofidx + dim] = 1.0; // weight
   //          }
   //       }
   //    }
   //    // Debugging
   //    // cout << "offset = " << eidx_offset << endl;
   //    // cout << "control_points :" << endl;
   //    // control_points.Print(mfem::out);
   //    // Create low-order patch
   //    // lo_patches[p] = new NURBSPatch(lo_kvs, pdim, control_points.GetData());
   // }

   // NURBSExtension ext(&patchTopo, patches);

   // // 3. Optionally, increase the NURBS degree.
   // if (nurbs_degree_increase>0)
   // {
   //    mesh.DegreeElevate(nurbs_degree_increase);
   // }

   // // 4. Refine the mesh to increase the resolution.
   // for (int l = 0; l < ref_levels; l++)
   // {
   //    mesh.NURBSUniformRefinement();
   // }

   // // Create the LOR mesh
   // Mesh lo_mesh = mesh.GetLowOrderNURBSMesh(sptype);

   // // Write to file
   // ofstream ofs("lo_mesh.mesh");
   // ofs.precision(8);
   // lo_mesh.Print(ofs);


   // ofstream orig_ofs("mesh.mesh");
   // orig_ofs.precision(8);
   // mesh.Print(orig_ofs);

   return 0;
}

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
   int interp_rule_ = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-d", "--dim",
                  "Dimension of the mesh (1, 2, or 3).");
   args.AddOption(&np, "-n", "--num-patches",
                  "Number of patches in the mesh per dimension.");
   args.AddOption(&order, "-o", "--order",
                  "Order of nurbs bases.");
   args.AddOption(&interp_rule_, "-interp", "--interpolation-rule",
                  "Interpolation Rule: 0 - Greville, 1 - Botella, 2 - Demko");
   args.Parse();

   // Print & verify options
   args.PrintOptions(cout);
   MFEM_ASSERT(dim >= 1 && dim <= 3, "Invalid dimension");
   MFEM_ASSERT(np >= 1, "Must have at least one patch");
   MFEM_ASSERT(order >= 1, "Order of nurbs bases must be at least 1");
   NURBSInterpolationRule interp_rule = static_cast<NURBSInterpolationRule>(interp_rule_);

   // 1. Parameters
   int nx = (dim >= 1) ? np : 1; // Number of patches in each dimension
   int ny = (dim >= 2) ? np : 1;
   int nz = (dim == 3) ? np : 1;
   const int NP = nx*ny*nz;      // Total number of patches in the mesh
   const int pdim = dim + 1;     // Projective/homogeneous dimension

   // 2. Create the patch-topology mesh
   //    Default ordering is space-filling-curve, set to false to get Cartesian ordering
   Mesh patchTopo;
   if (dim == 1)
   {
      // patchTopo = new Mesh::MakeCartesian1D(NP, L);
      patchTopo = Mesh::MakeCartesian1D(nx, (real_t)nx);
   }
   else if (dim == 2)
   {
      patchTopo = Mesh::MakeCartesian2D
      (
         nx, ny, Element::QUADRILATERAL, true,
         (real_t)nx, (real_t)ny, false
      );
   }
   else if (dim == 3)
   {
      patchTopo = Mesh::MakeCartesian3D
      (
         nx, ny, nz, Element::HEXAHEDRON,
         (real_t)nx, (real_t)ny, (real_t)nz, false
      );
   }
   else
   {
      MFEM_ABORT("Invalid dimension");
   }
   // patchTopo.FinalizeTopology();
   // patchTopo.Finalize(false, true);
   // patchTopo.CheckBdrElementOrientation();

   // 3. Create the reference knotvectors and control points
   //    for each patch (same in all dimensions)
   Array<const KnotVector*> kv_ref(np);
   std::vector<Array<real_t>> cpts_ref(np);
   Vector knots;  // knot values
   Vector x;      // physical coordinates to interpolate
   int nel;       // Number of elements
   int ncp;       // Number of control points
   for (int I = 0; I < np; I++)
   {
      nel = I + 1;
      // For a spline basis with C^{-1} continuity at ends and
      // C^{p-1} continuity at interior knots, NCP = order + NEL
      ncp = order + nel;

      // Define knot vectors
      knots.SetSize(nel+1);
      for (int i = 0; i < nel+1; i++)
      {
         knots[i] = (real_t)i / nel;
      }
      kv_ref[I] = new KnotVector(order, knots);

      // Debugging
      cout << endl << "Patch index = " << I << endl;
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
      Vector cpts(ncp);
      kv_ref[I]->GetInterpolant(x, NURBSInterpolationRule::Uniform, cpts);
      cpts_ref[I].CopyFrom(cpts.GetData());

      // Debugging
      cout << "cpts_ref[I] :" << endl;
      cpts_ref[I].Print(cout);
   }


   // 4. Create the patches
   Array<NURBSPatch*> patches(NP);
   int I,J,K; // patch indices
   int i,j,k; // dof indices
   int dofidx;
   Array<int> NCP(3); // number of control points per dim
   NCP = 1; // init

   for (int p = 0; p < NP; p++)
   {
      // Debugging
      cout << endl << "Creating patch = " << p << endl;

      Array<const KnotVector*> kvs(dim);
      I = p % nx;
      J = (p / nx) % ny;
      K = p / (ny * nx);
      int IJK[3] = {I,J,K};

      // Collect the knot vectors for this patch
      for (int d = 0; d < dim; d++)
      {
         kvs[d] = new KnotVector(*kv_ref[IJK[d]]);
         cout << "  kvs[" << d << "] = ";
         kvs[d]->Print(cout);
         NCP[d] = kvs[d]->GetNCP();
      }

      // Debugging
      cout << "  I,J,K = " << I << " " << J << " " << K << endl;
      cout << "  NCP = " << NCP[0] << " " << NCP[1] << " " << NCP[2] << endl;

      // Define the control points for this patch
      // The domain for each patch in physical space is [I, I+1] x [J, J+1] x [K, K+1]
      Array<real_t> control_points(pdim * NCP.Prod());
      for (int k = 0; k < NCP[2]; k++)
      {
         for (int j = 0; j < NCP[1]; j++)
         {
            for (int i = 0; i < NCP[0]; i++)
            {
               dofidx = i + j*NCP[0] + k*NCP[0]*NCP[1];
               int ijk[3] = {i,j,k};

               // Set the control points (+ weight) for the LO mesh
               for (int d = 0; d < dim; d++)
               {
                  control_points[pdim*dofidx + d] = cpts_ref[IJK[d]][ijk[d]];
               }
               control_points[pdim*dofidx + dim] = 1.0; // weight
            }
         }
      }
      cout << endl;

      // Create patch
      patches[p] = new NURBSPatch(kvs, pdim, control_points.GetData());

      // Debugging
      cout << "  Patch " << p << " : " << endl;
      patches[p]->Print(cout);
      cout << endl;
   }

   // Debugging - different order of patches
   Array<NURBSPatch*> newpatches(NP);
   newpatches[0] = patches[0];
   newpatches[1] = patches[2];
   newpatches[2] = patches[1];
   newpatches[3] = patches[3];

   // Crate the mesh
   NURBSExtension ext(&patchTopo, patches);
   Mesh mesh = Mesh(ext);

   // Write to file
   ofstream orig_ofs("mesh.mesh");
   orig_ofs.precision(8);
   mesh.Print(orig_ofs);

   // Debugging - write patchTopo to file
   // ofstream topo_ofs("topo.mesh");
   // topo_ofs.precision(8);
   // patchTopo.Print(topo_ofs);

   // // Create the LOR mesh
   Mesh lo_mesh = mesh.GetLowOrderNURBSMesh(interp_rule);
   ofstream ofs("lo_mesh.mesh");
   ofs.precision(8);
   lo_mesh.Print(ofs);


   return 0;
}

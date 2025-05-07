//               Demonstration of generating a NURBS mesh
//
// Compile with: make nurbs_lor
//
// Sample runs:  nurbs_lor -ref 2 -incdeg 3 -pc 1
//
// Description:  This example code solves a diffusion problem with
//               different preconditioners and records some stats.
//               Preconditioner (-pc) choices:
//                 - 0: No PC
//                 - 1: LOR AMG (uniform spacing)
//                 - 2: LOR AMG (greville abscissa)
//                 - 3: LOR AMG (botella abscissa)
//

#include "mfem.hpp"
#include <iostream>

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init();
   Hypre::Init();

   // 1. Parse command-line options.
   const char *mesh_file = "../../data/square-nurbs.mesh";
   int ref_levels = 0;
   int nurbs_degree_increase = 0;  // Elevate the NURBS mesh degree by this
   int projection_type = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-ref", "--refine",
                  "Number of uniform mesh refinements.");
   args.AddOption(&nurbs_degree_increase, "-incdeg", "--nurbs-degree-increase",
                  "Elevate NURBS mesh degree by this amount.");
   args.AddOption(&projection_type, "-proj", "--projection",
                  "Projection Type: 0 - Greville, 1 - Botella, 2 - Demko");
   args.Parse();

   // Print & verify options
   args.PrintOptions(cout);
   SplineProjectionType sptype = static_cast<SplineProjectionType>(projection_type);

   // 2. Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();

   // Verify mesh is valid for this problem
   MFEM_VERIFY(mesh.IsNURBS(), "Example is for NURBS meshes");
   MFEM_VERIFY(mesh.GetNodes(), "NURBS mesh must have nodes");

   // 3. Optionally, increase the NURBS degree.
   if (nurbs_degree_increase>0)
   {
      mesh.DegreeElevate(nurbs_degree_increase);
   }

   // 4. Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.NURBSUniformRefinement();
   }

   // Create the LOR mesh
   Mesh lo_mesh = mesh.GetLowOrderNURBSMesh(sptype);

   ofstream ofs("lo_mesh.mesh");
   ofs.precision(8);
   lo_mesh.Print(ofs);

   return 0;
}

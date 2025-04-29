//               IGA with LOR Preconditioning
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
// #include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

enum class SplineIntegrationRule { FULL_GAUSSIAN, REDUCED_GAUSSIAN, };
void SetPatchIntegrationRules(const Mesh &mesh,
                              const SplineIntegrationRule &splineRule,
                              BilinearFormIntegrator * bfi);


int main(int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init();
   Hypre::Init();

   // 1. Parse command-line options.
   const char *mesh_file = "../../data/cube-nurbs.mesh";
   bool pa = false;
   bool patchAssembly = false;
   int ref_levels = 0;
   int nurbs_degree_increase = 0;  // Elevate the NURBS mesh degree by this
   int preconditioner = 0;
   int visport = 19916;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&patchAssembly, "-patcha", "--patch-assembly", "-no-patcha",
                  "--no-patch-assembly", "Enable patch-wise assembly.");
   args.AddOption(&ref_levels, "-ref", "--refine",
                  "Number of uniform mesh refinements.");
   args.AddOption(&nurbs_degree_increase, "-incdeg", "--nurbs-degree-increase",
                  "Elevate NURBS mesh degree by this amount.");
   args.AddOption(&preconditioner, "-pc", "--preconditioner",
                  "Preconditioner: 0 - none, 1 - diagonal, 2 - LOR AMG");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();

   // Print & verify options
   args.PrintOptions(cout);
   MFEM_VERIFY(!(patchAssembly && !pa), "Patch assembly must be used with -pa");
   if (preconditioner != 0)
   {
      MFEM_VERIFY(nurbs_degree_increase > 0,
                  "LOR preconditioner requires degree increase");
   }

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

   // 5. Define a finite element space on the mesh.
   // Node ordering is important - right now, only works with byVDIM
   FiniteElementCollection * fec = mesh.GetNodes()->OwnFEC();
   cout << "fec order = " << fec->GetOrder() << endl;

   // FiniteElementSpace *fespace = new FiniteElementSpace(&mesh, mesh.NURBSext, fec);
   // cout << "Finite Element Collection: " << fec->Name() << endl;
   // const int Ndof = fespace->GetTrueVSize();
   // cout << "Number of finite element unknowns: " << Ndof << endl;
   // cout << "Number of elements: " << fespace->GetNE() << endl;
   // cout << "Number of patches: " << mesh.NURBSext->GetNP() << endl;

   // // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   // Array<int> ess_tdof_list, ess_bdr(mesh.bdr_attributes.Max());
   // ess_bdr = 1;
   // fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);


   // // 7. Set up the linear form b(.)
   // LinearForm b(fespace);
   // ConstantCoefficient one(1.0);
   // b.AddDomainIntegrator(new DomainLFIntegrator(one));
   // cout << "Assembling RHS ... " << flush;
   // b.Assemble();
   // cout << "done." << endl;

   // // 8. Define the solution vector x as a finite element grid function
   // GridFunction x(fespace);
   // x = 0.0;

   // // 9. Set up the bilinear form a(.,.)
   // DiffusionIntegrator *di = new DiffusionIntegrator(one);

   // if (patchAssembly)
   // {
   //    di->SetIntegrationMode(NonlinearFormIntegrator::Mode::PATCHWISE);
   //    // SetPatchIntegrationRules(mesh, SplineIntegrationRule::FULL_GAUSSIAN, di);
   // }

   // // 10. Assembly
   // StopWatch sw;
   // // sw.Start();

   // // Define and assemble bilinear form
   // cout << "Assembling a ... " << flush;
   // BilinearForm a(fespace);
   // if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   // a.AddDomainIntegrator(di);
   // a.Assemble();
   // cout << "done." << endl;

   // // Form linear system
   // cout << "Forming linear system ... " << flush;
   // OperatorPtr A;
   // Vector B, X;
   // a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   // cout << "done. " << "(size = " << fespace->GetTrueVSize() << ")" << endl;

   // // 11. Get the preconditioner
   // // We define solver here because SetOperator needs to be used before
   // // SetPreconditioner *if* we are using hypre
   // CGSolver solver(MPI_COMM_WORLD);
   // solver.SetOperator(*A);

   // // Create the LOR mesh
   // // Modify patches?


   // // Get patches
   // Array<const KnotVector*> kv(dim);
   // mesh.NURBSext->GetPatchKnotVectors(0, kv);
   // // Get greville points
   // Vector greville(kv[0]->GetNCP());
   // for (int i = 0; i < kv[0]->GetNCP(); i++) { greville[i] = kv[0]->GetGreville(i); }
   // // Print
   // cout << "Knots : "; kv[0]->Print(mfem::out);
   // cout << "Greville points : "; greville.Print(mfem::out, 32);



   return 0;
}

// For each patch, sets
void SetPatchIntegrationRules(const Mesh &mesh,
                              const SplineIntegrationRule &splineRule,
                              BilinearFormIntegrator * bfi)
{
   const int dim = mesh.Dimension();
   NURBSMeshRules * patchRules  = new NURBSMeshRules(mesh.NURBSext->GetNP(), dim);
   // Loop over patches and set a different rule for each patch.
   for (int p=0; p < mesh.NURBSext->GetNP(); ++p)
   {
      Array<const KnotVector*> kv(dim);
      mesh.NURBSext->GetPatchKnotVectors(p, kv);

      std::vector<const IntegrationRule*> ir1D(dim);
      // Construct 1D integration rules by applying the rule ir to each knot span.
      for (int i=0; i<dim; ++i)
      {
         if ( splineRule == SplineIntegrationRule::FULL_GAUSSIAN )
         {
            const int order = kv[i]->GetOrder();
            const IntegrationRule ir = IntRules.Get(Geometry::SEGMENT, 2*order);
            // ir1D[i] = IntegrationRule::ApplyToKnotIntervals(ir,*kv[i]);
            ir1D[i] = ir.ApplyToKnotIntervals(*kv[i]);
         }
         else if ( splineRule == SplineIntegrationRule::REDUCED_GAUSSIAN )
         {
            // ir1D[i] = IntegrationRule::GetIsogeometricReducedGaussianRule(*kv[i]);
            MFEM_ABORT("Unknown PatchIntegrationRule1D")
         }
         else
         {
            MFEM_ABORT("Unknown PatchIntegrationRule1D")
         }
      }

      patchRules->SetPatchRules1D(p, ir1D);
   }  // loop (p) over patches

   patchRules->Finalize(mesh);
   bfi->SetNURBSPatchIntRule(patchRules);
}
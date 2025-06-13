//               Testing NURBS LOR code snippets
//
// Compile with: make nurbs_lor
//

#include "mfem.hpp"
#include <iostream>

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   const int vdim = 1;

   const char *mesh_file = "ho_mesh.mesh";
   bool printX = false;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&printX, "-X", "--printX", "-noX", "--no-printX",
                  "Print the interpolation matrix.");
   args.Parse();
   // Print & verify options
   args.PrintOptions(cout);

   Mesh mesh(mesh_file, 1, 1);
   // Mesh mesh("ho_mesh.mesh");
   // Mesh lo_mesh("lo_mesh.mesh");

   // Create a GridFunction on the HO mesh
   FiniteElementCollection* fec = mesh.GetNodes()->OwnFEC();
   FiniteElementSpace fespace = FiniteElementSpace(&mesh, fec, vdim,
                                                   Ordering::byVDIM);
   const long Ndof = fespace.GetTrueVSize();
   cout << "Number of finite element unknowns: " << Ndof << endl;
   cout << "Number of elements: " << fespace.GetNE() << endl;
   cout << "Number of patches: " << mesh.NURBSext->GetNP() << endl;
   cout << "getndof: " << mesh.NURBSext->GetNDof() << endl;

   SparseMatrix* Rinv = new SparseMatrix(Ndof, Ndof);
   Mesh lo_mesh = mesh.GetLowOrderNURBSMesh(NURBSInterpolationRule::Botella, vdim, Rinv);
   Rinv->Finalize();
   if (printX)
   {
      ofstream X_ofs("X.txt");
      cout << "Printing matrix to X.txt" << endl;
      Rinv->ToDenseMatrix()->PrintMatlab(X_ofs);
      // Rinv->PrintMatlab(X_ofs);
   }
   cout << "Finished creating low-order mesh." << endl;
   // I think, this is the most correct interpolation, but it may not be ideal
   // for larger problems. Other options include
   //   - Form the LO -> HO interpolation matrix instead
   //   - Use a sparse factorization or iterative solve for R
   // matrix?
   // Another optimization here would be to invert the vdim==1 matrix and
   // construct the R matrix from that.
   // DenseMatrix* R = Rinv->ToDenseMatrix();
   // R->PrintMatlab();
   // R->Invert();

   GridFunction x(&fespace);
   x = 0.0;
   for (int i = 0; i < fespace.GetTrueVSize(); i++)
   {
      // x(i) = 100.0 - (i-20.0)*(i-20.0); // example function
      x(i) = 1.0 + i;
   }

   // Create a GridFunction on the LO mesh
   FiniteElementCollection* lo_fec = lo_mesh.GetNodes()->OwnFEC();
   FiniteElementSpace lo_fespace = FiniteElementSpace(&lo_mesh, lo_fec, vdim,
                                                      Ordering::byVDIM);
   GridFunction lo_x(&lo_fespace);
   // lo_x = 0.0;
   Rinv->Mult(x, lo_x);
   cout << "Finished creating low-order grid function." << endl;



   // ----- Write to file -----
   ofstream x_ofs("x.gf");
   x_ofs.precision(16);
   x.Save(x_ofs);

   ofstream lo_x_ofs("lo_x.gf");
   lo_x_ofs.precision(16);
   lo_x.Save(lo_x_ofs);

   // Apply LO -> HO interpolation matrix
   // GridFunction x_recon(&lo_fespace);
   // R->AddMult(lo_x, x_recon);
   // cout << "Finished applying LO -> HO matrix." << endl;
   // ofstream x_recon_ofs("x_recon.gf");
   // x_recon_ofs.precision(16);
   // x_recon.Save(x_recon_ofs);

   // ----- Test GetInterpolationMatrix -----
   // Build a patch from scratch
   // const int tdim = 2;
   // const int pdim = 2;
   // Array<const KnotVector*> kvs(tdim);
   // kvs[0] = new KnotVector(2, Vector({0.0, 1.0}));
   // kvs[1] = new KnotVector(1, Vector({0.0, 1.0}));
   // Vector control_points(
   // {
   //    0.0, 0.0, 1.0,
   //    1.0, 0.0, 1.0,
   //    2.0, 0.0, 1.0,
   //    0.0, 1.0, 1.0,
   //    1.0, 1.0, 1.0,
   //    2.0, 1.0, 1.0,
   // });
   // NURBSPatch patch(kvs, pdim, control_points.GetData());

   // // Define new knots
   // Array<Vector *> uknots;
   // uknots.SetSize(tdim);
   // uknots[0] = new Vector({0.4, 0.80});
   // uknots[1] = new Vector({0.3});

   // // ----- Get the interpolation matrix R -----
   // /** R should be a 4x6 matrix with values:
   //     0.252  0.336  0.112  0.108  0.144  0.048
   //     0.252  0.336  0.112  0.108  0.144  0.048
   //     0.028  0.224  0.448  0.012  0.096  0.192
   //     0.028  0.224  0.448  0.012  0.096  0.192
   //  */
   // SparseMatrix R(4,6);
   // int vdim = 2;
   // patch.GetInterpolationMatrix(R, uknots, vdim, 0, 0);
   // R.Finalize();

   // // Print as dense matrix
   // cout << "R = " << endl;
   // R.ToDenseMatrix()->PrintMatlab(cout);


   // // Free memory
   // for (int i = 0; i < tdim; i++)
   // {
   //    delete kvs[i];
   //    delete uknots[i];
   // }
   // delete R;

   return 0;
}

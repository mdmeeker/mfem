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
   const int vdim = 2;

   Mesh mesh("ho_mesh.mesh");
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

   SparseMatrix* R = new SparseMatrix(Ndof, Ndof);
   Mesh lo_mesh = mesh.GetLowOrderNURBSMesh(NURBSInterpolationRule::Botella, vdim,
                                            R);
   R->Finalize();

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
   lo_x = 0.0;

   // Test
   Array<int> vdofs;
   for (int p = 0; p < mesh.NURBSext->GetNP(); p++)
   {
      // Get the patch DOFs
      fespace.GetPatchVDofs(p, vdofs);
      cout << "Patch " << p << " DOFs: ";
      for (int i = 0; i < vdofs.Size(); i++)
      {
         cout << vdofs[i] << " ";
      }
      // Compare to nurbsext
      mesh.NURBSext->GetPatchDofs(p, vdofs);
      cout << " | NURBSPatch DOFs: ";
      for (int i = 0; i < vdofs.Size(); i++)
      {
         cout << vdofs[i] << " ";
      }
      cout << endl;
   }


   // ----- Test GetInterpolationMatrix -----
   // Interpolate the HO GridFunction onto the LO mesh
   // SparseMatrix R = mesh.GetNURBSInterpolationMatrix(lo_mesh, vdim);
   // SparseMatrix R = mesh.GetNURBSInterpolationMatrix(mesh, vdim);
   R->AddMult(x, lo_x);

   // Debugging
   // const int NP = mesh.NURBSext->GetNP();
   // const int dim = mesh.NURBSext->Dimension();
   // Array<int> nrows(NP);
   // Array<int> ncols(NP);
   // for (int p = 0; p < NP; p++)
   // {
   //    nrows[p] = vdim;
   //    ncols[p] = 1;
   //    for (int d = 0; d < dim; d++)
   //    {
   //       nrows[p] *= lo_mesh.NURBSext->GetKnotVector(d)->GetNUK();
   //       ncols[p] *= mesh.NURBSext->GetKnotVector(d)->GetNCP();
   //    }
   // }
   // mfem::out << "Mesh::GetNURBSInterpolationMatrix : " << endl;
   // mfem::out << "nrows = " << nrows.Sum()
   //           << ", ncols = " << ncols.Sum() << endl;
   // SparseMatrix R(nrows.Sum(), ncols.Sum());

   // // Use unique knots from target patch as interpolation points
   // Array<NURBSPatch*> patches(NP);
   // GetNURBSPatches(patches);
   // Array<NURBSPatch*> target_patches(NP);
   // mesh.GetNURBSPatches(target_patches);

   // // Build the interpolation matrix
   // int row_offset = 0;
   // int col_offset = 0;
   // for (int p = 0; p < NP; p++)
   // {
   //    patches[p]->GetInterpolationMatrix(*target_patches[p], R);
   //    row_offset += nrows[p];
   //    col_offset += ncols[p];
   // }

   // R.Finalize();


   // Print as dense matrix
   // cout << "R = " << endl;
   // R.ToDenseMatrix()->PrintMatlab(cout);

   // ----- Write to file -----
   ofstream x_ofs("x.gf");
   x_ofs.precision(16);
   x.Save(x_ofs);

   ofstream lo_x_ofs("lo_x.gf");
   lo_x_ofs.precision(16);
   lo_x.Save(lo_x_ofs);

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
   delete R;

   return 0;
}

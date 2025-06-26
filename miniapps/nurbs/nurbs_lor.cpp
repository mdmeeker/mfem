//               Testing NURBS LOR code snippets
//
// Compile with: make nurbs_lor
//

#include "mfem.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

// Class for applying the action of a Kronecker product using the
// Pot-RwCl algorithm
//
// Reference: Complexity of Memory-Efficient Kronecker Operations with
//            Applications to the Solution of Markov Models
class KroneckerProduct
{
private:
   Array<DenseMatrix*> A;
   int K;               // number of matrices
   Array<int> rows;     // sizes of each matrix
   Array<int> cols;
   int N;               // total number of rows
public:
   KroneckerProduct(const Array<DenseMatrix*> &A_) : A(A_)
   {
      K = A.Size();
      rows.SetSize(K);
      cols.SetSize(K);
      for (int k = 0; k < K; k++)
      {
         rows[k] = A[k]->Height();
         cols[k] = A[k]->Width();
      }
      N = rows.Prod();
   }

   void Mult(const Vector &x, Vector &y) const
   {
      MFEM_VERIFY(x.Size() == N, "Input vector must have size " << N);
      y.SetSize(N);
      y = 0.0;

      PotRwCl(K-1, 0, 0, 1.0, x, y);
   }

   void PotRwCl(int k, long long r, long long c, real_t value,
                         const Vector &x, Vector &y) const
   {
      const DenseMatrix &Ak = *A[k];
      for (int i = 0; i < rows[k]; i++)
      {
         for (int j = 0; j < cols[k]; j++)
         {
            if (Ak(i, j) == 0) { continue; }

            long long new_r = r * rows[k] + i;
            long long new_c = c * cols[k] + j;
            real_t new_value = value * Ak(i, j);

            if (k == 0)
            {
               y[new_r] += new_value * x[new_c];
            }
            else
            {
               PotRwCl(k - 1, new_r, new_c, new_value, x, y);
            }
         }
      }
   }
};


class NURBSInterpolator
{
private:
   Mesh* ho_mesh; // High-order mesh
   Mesh* lo_mesh; // Low-order mesh
   int vdim; // Vector dimension (default 1)
   int NP; // Number of patches
   int dim; // Topological dimension
   int ho_Ndof; // Number of dofs in HO mesh
   int lo_Ndof; // Number of dofs in LO mesh

   Array2D<SparseMatrix*> X; // transfer matrices from HO->LO, per patch/dimension
   Array2D<DenseMatrix*> R; // transfer matrices from LO->HO, per patch/dimension
   Array<KroneckerProduct*> kron; // Kronecker product actions for each patch

   std::vector<Array<int>> ho_p2g; // Patch to global mapping for HO mesh
   std::vector<Array<int>> lo_p2g; // Patch to global mapping for LO mesh

public:
   NURBSInterpolator(Mesh* ho_mesh_, Mesh* lo_mesh_, int vdim_ = 1) :
      ho_mesh(ho_mesh_),
      lo_mesh(lo_mesh_),
      vdim(vdim_),
      NP(ho_mesh->NURBSext->GetNP()),
      dim(ho_mesh->NURBSext->Dimension()),
      ho_Ndof(ho_mesh->NURBSext->GetNDof()),
      lo_Ndof(lo_mesh->NURBSext->GetNDof())
   {
      // Basic checks
      MFEM_VERIFY(ho_mesh->IsNURBS(), "HO mesh must be a NURBS mesh.")
      MFEM_VERIFY(lo_mesh->IsNURBS(), "LO mesh must be a NURBS mesh.")
      MFEM_VERIFY(NP == lo_mesh->NURBSext->GetNP(),
               "Meshes must have the same number of patches.");
      MFEM_VERIFY(dim == lo_mesh->NURBSext->Dimension(),
               "Meshes must have the same topological dimension.");

      // Collect X, R, and kron
      X.SetSize(NP, dim);
      R.SetSize(NP, dim);
      kron.SetSize(NP);
      for (int p = 0; p < NP; p++)
      {
         Array<const KnotVector*> ho_kvs(dim);
         Array<const KnotVector*> lo_kvs(dim);
         ho_mesh->NURBSext->GetPatchKnotVectors(p, ho_kvs);
         lo_mesh->NURBSext->GetPatchKnotVectors(p, lo_kvs);
         Vector u;
         for (int d = 0; d < dim; d++)
         {
            lo_kvs[d]->GetUniqueKnots(u);
            X(p, d) = new SparseMatrix(ho_kvs[d]->GetInterpolationMatrix(u));
            X(p, d)->Finalize();
            R(p, d) = new DenseMatrix(*X(p, d)->ToDenseMatrix());
            R(p, d)->Invert();
         }

         // Create classes for taking kron prod
         Array<DenseMatrix*> A(dim);
         R.GetRow(p, A);
         kron[p] = new KroneckerProduct(A);
      }

      // Collect patch to global mappings
      ho_p2g.resize(NP);
      lo_p2g.resize(NP);
      for (int p = 0; p < NP; p++)
      {
         ho_mesh->NURBSext->GetPatchDofs(p, ho_p2g[p]);
         lo_mesh->NURBSext->GetPatchDofs(p, lo_p2g[p]);
      }
   }

   // Apply R using kronecker product
   void ApplyR(const Vector &x, Vector &y)
   {
      Vector xp, yp;
      y.SetSize(ho_Ndof);
      y = 0.0;
      for (int p = 0; p < NP; p++)
      {
         x.GetSubVector(lo_p2g[p], xp);
         kron[p]->Mult(xp, yp);
         y.SetSubVector(ho_p2g[p], yp);
      }
      cout << "x = ";
      x.Print(cout, 100);
      cout << "y = ";
      y.Print(cout, 100);
   }

};

void Save(const char *filename, const GridFunction &x)
{
   ofstream ofs(filename);
   ofs.precision(16);
   x.Save(ofs);
   cout << "Saved vector to " << filename << endl;
}
void Save(const char *filename, DenseMatrix *A)
{
   ofstream ofs(filename);
   ofs.precision(16);
   A->PrintMatlab(ofs);
   cout << "Saved matrix to " << filename << endl;
}
void Save(const char *filename, SparseMatrix *A)
{
   if (!A->Finalized()) { A->Finalize(); }
   Save(filename, A->ToDenseMatrix());
}


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

   SparseMatrix* X = new SparseMatrix(Ndof, Ndof);
   Mesh lo_mesh = mesh.GetLowOrderNURBSMesh(NURBSInterpolationRule::Botella, vdim, X);
   if (printX) { Save("X.txt", X); }
   cout << "Finished creating low-order mesh." << endl;

   // Get dimension interpolation matrices: X1, X2, X3
   const int tdim = mesh.NURBSext->Dimension();
   Array<const KnotVector*> kvs(tdim);
   Array<const KnotVector*> lo_kvs(tdim);
   Array<Vector*> lo_uknots(tdim);
   mesh.NURBSext->GetPatchKnotVectors(0, kvs);
   lo_mesh.NURBSext->GetPatchKnotVectors(0, lo_kvs);

   mesh.NURBSext->AssembleCollocationMatrix(NURBSInterpolationRule::Botella);

   SparseMatrix X0 = kvs[0]->GetInterpolationMatrix(NURBSInterpolationRule::Botella);
   SparseMatrix X1 = kvs[1]->GetInterpolationMatrix(NURBSInterpolationRule::Botella);
   if (printX)
   {
      Save("X0.txt", &X0);
      Save("X1.txt", &X1);
   }

   // Create a NURBSInterpolator object
   NURBSInterpolator interpolator(&mesh, &lo_mesh);

   GridFunction ho_x(&fespace);
   ho_x = 0.0;
   for (int i = 0; i < fespace.GetTrueVSize(); i++)
   {
      // ho_x(i) = 100.0 - (i-10.0)*(i-10.0); // example function
      ho_x(i) = 1.0 + i;
   }

   // Create a GridFunction on the LO mesh
   FiniteElementCollection* lo_fec = lo_mesh.GetNodes()->OwnFEC();
   FiniteElementSpace lo_fespace = FiniteElementSpace(&lo_mesh, lo_fec, vdim,
                                                      Ordering::byVDIM);
   GridFunction lo_x(&lo_fespace);
   X->Mult(ho_x, lo_x);
   cout << "Finished creating low-order grid function." << endl;

   // Now compare with the results of NURBSInterpolator
   GridFunction x_recon(&fespace);

   interpolator.ApplyR(lo_x, x_recon);

   // ----- Write to file -----
   Save("ho_x.gf", ho_x);
   Save("lo_x.gf", lo_x);
   Save("x_recon.gf", x_recon);

   return 0;
}

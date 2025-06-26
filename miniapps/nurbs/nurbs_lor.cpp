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

   FiniteElementSpace* fespace; // Finite Element space for HO mesh

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

      // Finite Element space
      FiniteElementCollection* fec = ho_mesh->GetNodes()->OwnFEC();
      fespace = new FiniteElementSpace(ho_mesh, fec, vdim, Ordering::byVDIM);

      // Collect patch to global mappings
      ho_p2g.resize(NP);
      lo_p2g.resize(NP);
      for (int p = 0; p < NP; p++)
      {
         ho_mesh->NURBSext->GetPatchDofs(p, ho_p2g[p]);
         lo_mesh->NURBSext->GetPatchDofs(p, lo_p2g[p]);
      }
   }

   // Builds up the global transfer matrix: Xg[p] = kron(X[p,0], X[p,1], X[p,2]).
   // Generally this is pretty inefficient and is not needed other than for
   // troubleshooting. Instead, directly apply the action of R and R^T
   SparseMatrix GetXg() const
   {
      SparseMatrix Xg(ho_Ndof, lo_Ndof);
      for (int p = 0; p < NP; p++)
      {
         // Get Xp = X(p,0) kron X(p,1) kron X(p,2)
         SparseMatrix* Xp = nullptr;
         SparseMatrix* X01 = nullptr;
         SparseMatrix* X12 = nullptr;
         if (dim == 1)
         {
            SparseMatrix* Xp = new SparseMatrix(*X(p, 0));
            Xp->Finalize();
         }
         if (dim >= 2)
         {
            SparseMatrix* X01 = OuterProduct(*X(p, 1), *X(p, 0));
            X01->Finalize();
            Xp = X01;
         }
         if (dim == 3)
         {
            SparseMatrix* X12 = OuterProduct(*X(p, 2), *X01);
            X12->Finalize();
            Xp = X12;
            delete X01;
         }

         // Debugging
         // ofstream Xgp_ofs("Xgp.txt");
         // Xp->ToDenseMatrix()->PrintMatlab(Xgp_ofs);

         // Set values using patch -> global mapping
         Array<int> cols;
         Array<int> vcols;
         Vector srow;
         int rows = lo_p2g[p].Size();
         Array<int> dofs(ho_p2g[p]);

         for (int r = 0; r < rows; r++)
         {
            Xp->GetRow(r, cols, srow);

            for (int i=0; i<cols.Size(); ++i)
            {
               cols[i] = dofs[cols[i]];
            }

            for (int vd = 0; vd < vdim; vd++)
            {
               vcols = cols;
               fespace->DofsToVDofs(vd, vcols);
               int vdrow = fespace->DofToVDof(dofs[r], vd);
               Xg.SetRow(vdrow, vcols, srow);
            }
         }
      }
      Xg.Finalize();
      return Xg;
   }

   // Apply R using kronecker product
   void ApplyR(const Vector &x, Vector &y)
   {
      Vector xp, yp;
      y.SetSize(ho_Ndof);
      y = 0.0;
      for (int p = 0; p < NP; p++)
      {
         Array<int> vdofs;
         fespace->GetPatchVDofs(p, vdofs);

         x.GetSubVector(vdofs, xp);
         kron[p]->Mult(xp, yp);

         y.SetSubVector(vdofs, yp);
      }
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

   const int NP = 1;
   Array<NURBSPatch*> patches(NP);
   mesh.GetNURBSPatches(patches);
   Array<NURBSPatch*> lo_patches(NP);
   lo_mesh.GetNURBSPatches(lo_patches);

   SparseMatrix* X01 = OuterProduct(X0,X1);
   Save("X01.txt", X01);

   // Create a NURBSInterpolator object
   NURBSInterpolator interpolator(&mesh, &lo_mesh);

   SparseMatrix Xg = interpolator.GetXg();
   Save("Xg.txt", &Xg);

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

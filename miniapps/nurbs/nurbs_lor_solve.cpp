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
//                 - 1: LOR AMG (R = Identity)
//                 - 2: LOR AMG (choose interpolation with -interp)
//

#include "mfem.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

#include "linalg/dtensor.hpp"

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
   int M;               // total number of cols
   // Bandwidths for each matrix (0 is unset)
   Array<int> BW;
public:
   KroneckerProduct(const Array<DenseMatrix*> &A) : A(A)
   {
      K = A.Size();
      MFEM_VERIFY((K==2) || (K==3), "Must be 2 or 3 dimensions")
      rows.SetSize(K);
      cols.SetSize(K);
      BW.SetSize(K);
      for (int k = 0; k < K; k++)
      {
         rows[k] = A[k]->Height();
         cols[k] = A[k]->Width();
         BW[k] = rows[k];
      }
      N = rows.Prod();
      M = cols.Prod();
   }

   void Mult(const Vector &x, Vector &y) const
   {
      MFEM_VERIFY(x.Size() == M, "Input vector must have size " << M);
      y.SetSize(N);
      y = 0.0;
      if (K == 2)
      {
         SumFactor2D<false>(x, y);
      }
      else if (K == 3)
      {
         SumFactor3D<false>(x, y);
      }
   }

   void MultTranspose(const Vector &x, Vector &y) const
   {
      MFEM_VERIFY(x.Size() == N, "Input vector must have size " << N);
      y.SetSize(M);
      y = 0.0;
      if (K == 2)
      {
         SumFactor2D<true>(x, y);
      }
      else if (K == 3)
      {
         SumFactor3D<true>(x, y);
      }

   }

   template<bool transpose = false>
   void SumFactor2D(const Vector &Xv, Vector &Yv) const
   {
      // Outer/inner loop sizes
      int OY = transpose ? rows[1] : cols[1];
      int OX = transpose ? rows[0] : cols[0];
      int IY = transpose ? cols[1] : rows[1];
      int IX = transpose ? cols[0] : rows[0];
      // Accumulator(s)
      Vector sumX(IX);
      // Reshape
      const auto X = Reshape(Xv.HostRead(), OX, OY);
      const auto Y = Reshape(Yv.HostReadWrite(), IX, IY);
      // Aliases
      const DenseMatrix &Ax = *A[0];
      const DenseMatrix &Ay = *A[1];
      const int bwx = std::floor(BW[0] / 2.0); // One-sided bandwidth
      const int bwy = std::floor(BW[1] / 2.0);

      for (int oy = 0; oy < OY; oy++)
      {
         sumX = 0.0;
         for (int ox = 0; ox < OX; ox++)
         {
            const real_t x = X(ox, oy);
            const int min_ix = std::max(0, ox - bwx);
            const int max_ix = std::min(IX, ox + bwx + 1);
            for (int ix = min_ix; ix < max_ix; ix++)
            {
               const real_t ax = transpose ? Ax(ox, ix) : Ax(ix, ox);
               sumX(ix) += x * ax;
            }
         }
         const int min_iy = std::max(0, oy - bwy);
         const int max_iy = std::min(IY, oy + bwy + 1);
         for (int iy = min_iy; iy < max_iy; iy++)
         {
            const real_t y = transpose ? Ay(oy, iy) : Ay(iy, oy);
            for (int ix = 0; ix < IX; ix++)
            {
               Y(ix, iy) += sumX(ix) * y;
            }
         }
      }
   }

   template<bool transpose = false>
   void SumFactor3D(const Vector &Xv, Vector &Yv) const
   {
      // Outer/inner loop sizes
      int OZ = transpose ? rows[2] : cols[2];
      int OY = transpose ? rows[1] : cols[1];
      int OX = transpose ? rows[0] : cols[0];
      int IZ = transpose ? cols[2] : rows[2];
      int IY = transpose ? cols[1] : rows[1];
      int IX = transpose ? cols[0] : rows[0];
      // Accumulator(s)
      Vector sumXYv(IX*IY);
      Vector sumX(IX);
      // Reshape
      const auto X = Reshape(Xv.HostRead(), OX, OY, OZ);
      const auto Y = Reshape(Yv.HostReadWrite(), IX, IY, IZ);
      auto sumXY = Reshape(sumXYv.HostReadWrite(), OX, OY);
      // Aliases
      const DenseMatrix &Ax = *A[0];
      const DenseMatrix &Ay = *A[1];
      const DenseMatrix &Az = *A[2];
      const int bwx = std::floor(BW[0] / 2.0); // One-sided bandwidth
      const int bwy = std::floor(BW[1] / 2.0);
      const int bwz = std::floor(BW[2] / 2.0);

      for (int oz = 0; oz < OZ; oz++)
      {
         sumXYv = 0.0;
         for (int oy = 0; oy < OY; oy++)
         {
            sumX = 0.0;
            // oz, oy, ox, ix
            for (int ox = 0; ox < OX; ox++)
            {
               const real_t x = X(ox, oy, oz);
               const int min_ix = std::max(0, ox - bwx);
               const int max_ix = std::min(IX, ox + bwx + 1);
               for (int ix = min_ix; ix < max_ix; ix++)
               {
                  const real_t ax = transpose ? Ax(ox, ix) : Ax(ix, ox);
                  sumX(ix) += x * ax;
               }
            }
            // oz, oy, iy, ix
            const int min_iy = std::max(0, oy - bwy);
            const int max_iy = std::min(IY, oy + bwy + 1);
            for (int iy = min_iy; iy < max_iy; iy++)
            {
               const real_t y = transpose ? Ay(oy, iy) : Ay(iy, oy);
               for (int ix = 0; ix < IX; ix++)
               {
                  sumXY(ix,iy) += sumX(ix) * y;
               }
            }
         } // for (oy)

         // oz, iz, iy, ix
         const int min_iz = std::max(0, oz - bwz);
         const int max_iz = std::min(IZ, oz + bwz + 1);
         for (int iz = min_iz; iz < max_iz; iz++)
         {
            const real_t z = transpose ? Az(oz, iz) : Az(iz, oz);
            for (int iy = 0; iy < IY; iy++)
            {
               for (int ix = 0; ix < IX; ix++)
               {
                  Y(ix, iy, iz) += sumXY(ix, iy) * z;
               }
            }
         }
      } // for (oz)
   }

   DenseMatrix GetBanded(const DenseMatrix& M, int bw)
   {
      const int I = M.Height();
      const int J = M.Width();
      MFEM_VERIFY((bw > 0) && (bw <= J), "Invalid bandwidth")
      int bwx = std::floor(bw / 2.0);
      DenseMatrix banded(I, J);
      banded = 0.0;

      for (int i = 0; i < I; i++)
      {
         const int min_j = std::max(0, i - bwx);
         const int max_j = std::min(J, i + bwx + 1);
         for (int j = min_j; j < max_j; j++)
         {
            banded(i, j) = M(i, j);
         }
      }
      // Debugging
      cout << "original matrix norm = " << M.FNorm2() << endl;
      cout << "banded matrix (bw " << bw << ") norm = " << banded.FNorm2() << endl;
      return banded;
   }

   // Computes the quantity
   // || A[k] * banded(A[k]^-1, bw) - I ||_{F2} / || A[k] ||_{F2}
   // which is a metric for how well the banded inverse approximates
   // the full inverse.
   real_t BandedInverseNorm(int k, int bw)
   {
      const DenseMatrix &Ak = *A[k];
      MFEM_VERIFY(rows[k] == cols[k], "Only square matrices");
      const DenseMatrix Akb = GetBanded(Ak, bw);
      DenseMatrix Akbi(Akb);
      Akbi.Invert();
      // numerator = Ak * Akb.Inverse() - I
      DenseMatrix numerator(rows[k], rows[k]);
      mfem::Mult(Ak, Akbi, numerator);
      for (int i = 0; i < rows[k]; i++)
      {
         numerator(i, i) -= 1.0;
      }

      return numerator.FNorm2() / Ak.FNorm2();
   }

   // Set bandwidth for kth matrix such that BandedIvnerseNorm < tol
   void SetBandwidth(int k, real_t tol = 1e-5, int bw0 = 1)
   {
      MFEM_VERIFY((k >= 0) && (k < K), "Invalid matrix index");
      for (int bw = bw0; bw <= cols[k]; bw += 2)
      {
         real_t norm = BandedInverseNorm(k, bw);
         if (norm < tol)
         {
            BW[k] = bw;
            cout << "Set bandwidth for index " << k << " to " << bw
                 << ". norm = " << norm << endl;
            return;
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

   Array2D<DenseMatrix*> R; // transfer matrices from LO->HO, per patch/dimension
   std::vector<Array<int>> ho_p2g; // Patch to global mapping for HO mesh
   std::vector<Array<int>> lo_p2g; // Patch to global mapping for LO mesh

   Array2D<int> orders; // Order of basis per patch/dimension

public:
   Array<KroneckerProduct*> kron; // Kronecker product actions for each patch

   NURBSInterpolator(Mesh* ho_mesh, Mesh* lo_mesh, real_t threshold = 0.0, int vdim = 1) :
      ho_mesh(ho_mesh),
      lo_mesh(lo_mesh),
      vdim(vdim),
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

      // Collect R, and kron
      R.SetSize(NP, dim);
      orders.SetSize(NP, dim);
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
            orders(p, d) = ho_kvs[d]->GetOrder();
            lo_kvs[d]->GetUniqueKnots(u);
            SparseMatrix X(ho_kvs[d]->GetInterpolationMatrix(u));
            X.Finalize();
            R(p, d) = new DenseMatrix(*X.ToDenseMatrix());
            R(p, d)->Invert();
         }

         // Create classes for taking kron prod
         Array<DenseMatrix*> A(dim);
         R.GetRow(p, A);
         kron[p] = new KroneckerProduct(A);
      }
      if (threshold > 0)
      {
         SetBandwidths(threshold, false);
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
   void Mult(const Vector &x, Vector &y)
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
   }

   // Apply R^T using kronecker product
   void MultTranspose(const Vector &x, Vector &y)
   {
      Vector xp, yp;
      y.SetSize(lo_Ndof);
      y = 0.0;
      for (int p = 0; p < NP; p++)
      {
         x.GetSubVector(ho_p2g[p], xp);
         kron[p]->MultTranspose(xp, yp);
         y.SetSubVector(lo_p2g[p], yp);
      }
   }

   // bw0 = order*2 + 1 is a good initial guess
   void SetBandwidths(real_t tol = 1.0e-5, bool guess_bw0 = true)
   {
      for (int p = 0; p < NP; p++)
      {
         for (int d = 0; d < dim; d++)
         {
            const int bw0 = (guess_bw0) ? (orders(p, d) * 2 + 1) : 1;
            kron[p]->SetBandwidth(d, tol, bw0);
         }
      }
   }

};

enum class SplineIntegrationRule { FULL_GAUSSIAN, REDUCED_GAUSSIAN, };
void SetPatchIntegrationRules(const Mesh &mesh,
                              const SplineIntegrationRule &splineRule,
                              BilinearFormIntegrator * bfi);

class R_A_Rt : public Operator
{
private:
   // const Operator *R;
   NURBSInterpolator *R;
   const Operator *A;
   mutable Vector t1, t2;
   MemoryClass mem_class;

public:
   R_A_Rt(NURBSInterpolator *R, const Operator *A)
   : Operator(A->Height(), A->Width())
   , R(R), A(A)
   {
      mem_class = A->GetMemoryClass();//*C->GetMemoryClass();
      MemoryType mem_type = GetMemoryType(mem_class);
      t1.SetSize(A->Height(), mem_type);
      t2.SetSize(A->Height(), mem_type);
   }

   MemoryClass GetMemoryClass() const override { return mem_class; }

   void Mult(const Vector &x, Vector &y) const override
   { R->MultTranspose(x, t1); A->Mult(t1, t2); R->Mult(t2, y); }

   // ~R_A_Rt() override;
};

class NURBSLORPreconditioner : public Solver
{
private:
   NURBSInterpolator* R;   // transfer operator from LO->HO
   const Operator* A;      // AMG on LO dofs

   R_A_Rt* op;
   const ConstrainedOperator* opcon;

public:
   NURBSLORPreconditioner(
      NURBSInterpolator* R,
      const Array<int> & ess_tdof_list,
      const HypreBoomerAMG* A)
   : Solver(A->Height(), A->Width(), false)
   , R(R), A(A)
   {
      op = new R_A_Rt(R, A);
      opcon = new ConstrainedOperator(op, ess_tdof_list);
   }

   // y = P x = R A^-1 R^T x
   void Mult(const Vector &x, Vector &y) const
   {
      y = 0.0;
      opcon->Mult(x, y);
   }

   void SetOperator(const Operator &op) { A = &op;};

};

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
   int interp_rule_ = 1;
   int preconditioner = 0;
   bool visualization = false;

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
   args.AddOption(&interp_rule_, "-int", "--interpolation-rule",
                  "Interpolation rule: 0 - Greville, 1 - Botella, 2 - Demko, 3 - Uniform");
   args.AddOption(&preconditioner, "-pc", "--preconditioner",
                  "Preconditioner: 0 - none, 1 - diagonal, 2 - LOR AMG");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();

   // Print & verify options
   args.PrintOptions(cout);
   MFEM_VERIFY(!(patchAssembly && !pa), "Patch assembly must be used with -pa");
   NURBSInterpolationRule interp_rule = static_cast<NURBSInterpolationRule>
                                        (interp_rule_);

   // 2. Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();
   const int NP = mesh.NURBSext->GetNP();

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
   FiniteElementCollection* fec = mesh.GetNodes()->OwnFEC();
   FiniteElementSpace fespace = FiniteElementSpace(&mesh, fec);

   const int Ndof = fespace.GetTrueVSize();
   cout << "Number of finite element unknowns: " << Ndof << endl;
   cout << "Number of elements: " << fespace.GetNE() << endl;
   cout << "Number of patches: " << NP << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> ess_tdof_list, ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   // mesh.MarkExternalBoundaries(ess_bdr);
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 7. Define the solution vector x as a finite element grid function
   GridFunction x(&fespace);
   x = 0.0;

   // 8. Setup linear form b(.)
   ConstantCoefficient one(1.0);
   LinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 9. Setup bilinear form a(.,.)
   DiffusionIntegrator *di = new DiffusionIntegrator(one);

   if (patchAssembly)
   {
      di->SetIntegrationMode(NonlinearFormIntegrator::Mode::PATCHWISE);
      SetPatchIntegrationRules(mesh, SplineIntegrationRule::FULL_GAUSSIAN, di);
   }

   // 10. Assembly
   StopWatch sw;
   sw.Start();

   // Define and assemble bilinear form
   cout << "Assembling a ... " << flush;
   BilinearForm a(&fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.AddDomainIntegrator(di);
   a.Assemble();
   cout << "done." << endl;

   // Form linear system
   cout << "Forming linear system ... " << flush;
   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   cout << "done. " << "(size = " << fespace.GetTrueVSize() << ")" << endl;

   // 11. Get the preconditioner
   // We define solver here because SetOperator needs to be used before
   // SetPreconditioner *if* we are using hypre
   CGSolver solver(MPI_COMM_WORLD);
   solver.SetOperator(*A);

   // No preconditioner
   if (preconditioner == 0)
   {
      cout << "No preconditioner set ... " << endl;
   }
   // LOR AMG
   else if ((preconditioner == 1) || (preconditioner == 2))
   {
      cout << "Setting up preconditioner (LOR AMG) ... " << endl;

      // Create the LOR mesh
      const int vdim = fespace.GetVDim();
      Mesh lo_mesh = mesh.GetLowOrderNURBSMesh(interp_rule, vdim, NULL);

      // Write low order mesh to file
      if (visualization)
      {
         ofstream ofs("lo_mesh.mesh");
         ofs.precision(8);
         lo_mesh.Print(ofs);
      }

      FiniteElementCollection* lo_fec = lo_mesh.GetNodes()->OwnFEC();
      cout << "lo_fec order: " << lo_fec->GetOrder() << endl;
      FiniteElementSpace lo_fespace = FiniteElementSpace(&lo_mesh, lo_fec);
      const int lo_Ndof = lo_fespace.GetTrueVSize();
      MFEM_VERIFY(Ndof == lo_Ndof, "Low-order problem requires same Ndof");

      Array<int> lo_ess_tdof_list, lo_ess_bdr(lo_mesh.bdr_attributes.Max());
      lo_ess_bdr = 1;
      // lo_mesh.MarkExternalBoundaries(lo_ess_bdr);
      lo_fespace.GetEssentialTrueDofs(lo_ess_bdr, lo_ess_tdof_list);

      GridFunction lo_x(&lo_fespace);
      lo_x = 0.0;

      LinearForm lo_b(&lo_fespace);
      lo_b.AddDomainIntegrator(new DomainLFIntegrator(one));
      lo_b.Assemble();

      DiffusionIntegrator *lo_di = new DiffusionIntegrator(one);

      // Set up problem
      BilinearForm lo_a(&lo_fespace);
      lo_a.AddDomainIntegrator(lo_di);
      lo_a.Assemble();

      // Define linear system
      OperatorPtr lo_A;
      Vector lo_B, lo_X;
      lo_a.FormLinearSystem(lo_ess_tdof_list, lo_x, lo_b, lo_A, lo_X, lo_B);
      // Array<int> nobcs;
      // lo_a.FormLinearSystem(nobcs, lo_x, lo_b, lo_A, lo_X, lo_B);

      // Set up HypreBoomerAMG on the low-order problem
      HYPRE_BigInt row_starts[2] = {0, Ndof};
      SparseMatrix *lo_Amat = new SparseMatrix(lo_a.SpMat());
      HypreParMatrix *lo_A_hypre = new HypreParMatrix(
         MPI_COMM_WORLD,
         HYPRE_BigInt(Ndof),
         row_starts,
         lo_Amat
      );
      HypreBoomerAMG *lo_P = new HypreBoomerAMG(*lo_A_hypre);

      if (preconditioner == 1)
      {
         cout << "LOR AMG: R = Identity ... " << endl;
         solver.SetPreconditioner(*lo_P);
      }
      else
      {
         cout << "LOR AMG: R = X^-1 ... " << endl;

         NURBSInterpolator* interpolator = new NURBSInterpolator(&mesh, &lo_mesh, 1e-5);
         // interpolator->SetBandwidths(1e-4); // uses initial guess of order*2+1
         NURBSLORPreconditioner *P = new NURBSLORPreconditioner(interpolator, ess_tdof_list, lo_P);
         solver.SetPreconditioner(*P);
      }
   }
   else
   {
      MFEM_ABORT("Invalid preconditioner setting.")
   }

   sw.Stop();
   const real_t timeAssemble = sw.RealTime();
   sw.Clear();
   sw.Start();

   // 12. Solve the linear system A X = B.
   cout << "Solving linear system ... " << endl;
   solver.SetMaxIter(1e5);
   solver.SetPrintLevel(1);
   // These tolerances end up getting squared
   solver.SetRelTol(1e-8);
   solver.SetAbsTol(0);

   solver.Mult(B, X);

   cout << "Done solving system." << endl;
   sw.Stop();
   const real_t timeSolve = sw.RealTime();
   const real_t timeTotal = timeAssemble + timeSolve;

   // Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   // 13. Collect results and write to file
   const long Niter = solver.GetNumIterations();
   const long dof_per_sec_solve = (long)Ndof * Niter / timeSolve;
   const long dof_per_sec_total = (long)Ndof * Niter / timeTotal;
   cout << "Time to assemble: " << timeAssemble << " seconds" << endl;
   cout << "Time to solve: " << timeSolve << " seconds" << endl;
   cout << "Total time: " << timeTotal << " seconds" << endl;
   cout << "Dof/sec (solve): " << dof_per_sec_solve << endl;
   cout << "Dof/sec (total): " << dof_per_sec_total << endl;

   ofstream results_ofs("nurbs_lor_solve_results.csv", ios_base::app);
   // If file does not exist, write the header
   if (results_ofs.tellp() == 0)
   {
      results_ofs << "patcha, pa, pc, proj, "         // settings
                  << "mesh, refs, deg_inc, ndof, "    // mesh
                  << "niter, absnorm, relnorm, "      // solver
                  << "linf, l2, "                     // solution
                  << "t_assemble, t_solve, t_total, " // timing
                  << "dof/s_solve, dof/s_total"       // benchmarking
                  << endl;
   }

   results_ofs << patchAssembly << ", "               // settings
               << pa << ", "
               << preconditioner << ", "
               << interp_rule_ << ", "
               << mesh_file << ", "                   // mesh
               << ref_levels << ", "
               << nurbs_degree_increase << ", "
               << Ndof << ", "
               << Niter << ", "                       // solver
               << solver.GetFinalNorm() << ", "
               << solver.GetFinalRelNorm() << ", "
               << x.Normlinf() << ", "                // solution
               << x.Norml2() << ", "
               << timeAssemble << ", "                // timing
               << timeSolve << ", "
               << timeTotal << ", "
               << dof_per_sec_solve << ", "           // benchmarking
               << dof_per_sec_total << endl;

   results_ofs.close();

   // 14. Save the mesh and the solution
   if (visualization)
   {
      cout << "Saving mesh and solution to file..." << endl;
      ofstream mesh_ofs("mesh.mesh");
      mesh_ofs.precision(8);
      mesh.Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(16);
      x.Save(sol_ofs);
   }

   // 15. Free the used memory.
   // delete R;

   return 0;
}

// For each patch, sets integration rules
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
         const int order = kv[i]->GetOrder();

         if ( splineRule == SplineIntegrationRule::FULL_GAUSSIAN )
         {
            const IntegrationRule* ir = &IntRules.Get(Geometry::SEGMENT, 2*order);
            ir1D[i] = ir->ApplyToKnotIntervals(*kv[i]);
            // ir1D[i] = IntegrationRule::ApplyToKnotIntervals(ir,*kv[i]);
         }
         // else if ( splineRule == SplineIntegrationRule::REDUCED_GAUSSIAN )
         // {
         //    // ir1D[i] = IntegrationRule::GetIsogeometricReducedGaussianRule(*kv[i]);
         //    MFEM_ABORT("Unknown PatchIntegrationRule1D")
         // }
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
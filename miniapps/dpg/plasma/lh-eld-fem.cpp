//                   MFEM FEM parallel example
//

// Electron Landau Damping
// Strong formulation:
//      ∇×(1/μ₀ ∇×E) - ω² ϵ₀ ϵᵣ E - i ω²ϵ₀(Jₕ⁽¹⁾ + Jₕ⁽²⁾) = 0,   in Ω
//                    - Δ∥ Jₕ⁽¹⁾ + c₁ Jₕ⁽¹⁾ - c₁ P(r) E∥ = 0,   in Ω     
//                    - Δ∥ Jₕ⁽²⁾ + c₂ Jₕ⁽²⁾ + c₂ P(r) E∥ = 0,   in Ω 
//                                                  E×n = E₀,  on ∂Ω
// weak formulation:
//   Find E ∈ H(curl,Ω), Jₕ⁽¹⁾ ∈ H¹(Ω), Jₕ⁽²⁾ ∈ H¹(Ω) such that
//   (1/μ₀ ∇×E, ∇ × F) - ω² ϵ₀ (ϵᵣ E, F) - i ω²ϵ₀( (Jₕ⁽¹⁾ + Jₕ⁽²⁾), F) = 0,  ∀  F ∈ H(curl,Ω)
//      (b ⊗ b ∇Jₕ⁽¹⁾ : ∇G) + c₁ (Jₕ⁽¹⁾, G) - c₁ (P(r) (b ⊗ b) E, G) = 0,  ∀ G ∈ (H¹(Ω))ᵈ 
//      (b ⊗ b ∇Jₕ⁽²⁾ : ∇H) + c₂ (Jₕ⁽²⁾, H) + c₂ (P(r) (b ⊗ b) E, H) = 0,  ∀ H ∈ (H¹(Ω))ᵈ

#include "mfem.hpp"
#include "../util/pcomplexweakform.hpp"
#include "../util/pcomplexblockform.hpp"
#include "../util/utils.hpp"
#include "../util/maxwell_utils.hpp"
#include "../../common/mfem-common.hpp"
#include <fstream>
#include <iostream>
#include <cstring>
#include <filesystem>

using namespace std;
using namespace mfem;

real_t delta = 0.01; 
real_t a0 = -1.0;    
real_t a1 = 5.0;     

real_t pfunc_r(const Vector &x)
{
   real_t r = std::sqrt(x(0) * x(0) + x(1) * x(1));
   return a0 + a1 *(r-0.9);
}

real_t pfunc_i(const Vector &x)
{
   return delta;   
}

real_t sfunc_r(const Vector &x)
{
   return 1.0;
}

real_t sfunc_i(const Vector &x)
{
   return delta;
}

void bfunc(const Vector &x, Vector &b)
{
   real_t r = std::sqrt(x(0) * x(0) + x(1) * x(1));
   int dim = x.Size();
   b.SetSize(dim); b = 0.0;
   b(0) = -x(1) / r;
   b(1) =  x(0) / r;
   if (dim == 3) b(2) = 0.0;
}

void bcrossb(const Vector &x, DenseMatrix &bb)
{
   Vector b;
   bfunc(x, b);
   bb.SetSize(b.Size());
   MultVVt(b, bb);
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "data/LH_hot.msh";

   int order = 2;
   int par_ref_levels = 0;
   bool visualization = false;
   // real_t rnum=4.6e9;
   // real_t mu = 1.257e-6;
   // real_t eps0 = 8.8541878128e-12*factor;
   real_t rnum=1.5e9;
   real_t mu = 1.257e-6;
   real_t eps0 = 8.8541878128e-12;
   bool eld = false; // enable/disable electron Landau damping 
   real_t c1 = 25e6; 
   real_t c2 = 1e6; 
   bool paraview = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&par_ref_levels, "-pr", "--parallel-refinement_levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&rnum, "-rnum", "--number_of_wavelenths",
                  "Number of wavelengths");
   args.AddOption(&mu, "-mu", "--permeability",
                  "Permeability of free space (or 1/(spring constant)).");
   args.AddOption(&a0, "-a0", "--a0", "P(r) first parameter.");
   args.AddOption(&a1, "-a1", "--a1", "P(r) second parameter.");
   args.AddOption(&delta, "-delta", "--delta", "stability parameter.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&eld, "-eld", "--eld", "-no-eld",
                  "--no-eld",
                  "Enable or disable electron Landau damping.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   real_t omega = 2.*M_PI*rnum;
   if (eld) delta = 0.0; // disable delta if electron Landau damping is enabled

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   Array<int> int_bdr_attr;
   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      if (mesh.FaceIsInterior(mesh.GetBdrElementFaceIndex(i)))
      {
         int_bdr_attr.Append(mesh.GetBdrAttribute(i));
      }
   }
   int_bdr_attr.Sort();
   int_bdr_attr.Unique();

   // mesh.RemoveInternalBoundaries();
   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   int nattr = pmesh.attributes.Max();
   Array<int> attr(nattr);
   for (int i = 0; i<nattr; i++) { attr[i] = i+1; }
   
   ConstantCoefficient one_cf(1.0);
   ConstantCoefficient zero_cf(0.0);
   Array<Coefficient*> coeff_array_one(nattr);
   Array<Coefficient*> coeff_array_zero(nattr);

   DenseMatrix Mone(dim); 
   Mone = 0.0; Mone(0,0) = Mone(1,1) = 1.0;
   MatrixConstantCoefficient Mone_cf(Mone);
   DenseMatrix Mzero(dim); Mzero = 0.0;
   MatrixConstantCoefficient Mzero_cf(Mzero);

   Array<MatrixCoefficient*> coefs_r(nattr);
   Array<MatrixCoefficient*> coefs_i(nattr);
   for (int i = 0; i < nattr-1; ++i)
   {
      coefs_r[i] = &Mone_cf;
      coefs_i[i] = &Mzero_cf;
   }

   FunctionCoefficient S_cf_r(sfunc_r);
   FunctionCoefficient S_cf_i(sfunc_i);
   FunctionCoefficient P_cf_r(pfunc_r);
   FunctionCoefficient P_cf_i(pfunc_i);

   MatrixFunctionCoefficient bb_cf(dim,bcrossb);
   MatrixSumCoefficient oneminusbb(Mone_cf, bb_cf, 1.0, -1.0);
   ScalarMatrixProductCoefficient Soneminusbb_r(S_cf_r, oneminusbb);
   ScalarMatrixProductCoefficient Soneminusbb_i(S_cf_i, oneminusbb);

   ScalarMatrixProductCoefficient P_cf_bb_r(P_cf_r, bb_cf);
   ScalarMatrixProductCoefficient P_cf_bb_i(P_cf_i, bb_cf);

   MatrixSumCoefficient eps_r(Soneminusbb_r, P_cf_bb_r, 1.0, 1.0);
   MatrixSumCoefficient eps_i(Soneminusbb_i, P_cf_bb_i, 1.0, 1.0);

   coefs_r[nattr-1] = &eps_r;
   coefs_i[nattr-1] = &eps_i;

   PWMatrixCoefficient eps_cf_r(dim, attr, coefs_r);
   PWMatrixCoefficient eps_cf_i(dim, attr, coefs_i);

   for (int i = 0; i<par_ref_levels; i++)
   {
      pmesh.UniformRefinement();
      // eps_r_cf.Update();
      // eps_i_cf.Update();
   }

   ConstantCoefficient negeps0omeg2(-eps0 * omega * omega);

   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *E_fes = new ParFiniteElementSpace(&pmesh, fec);

   // Bilinear form coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient muinv(1./mu);

   Array<ParFiniteElementSpace *> pfes;
   pfes.Append(E_fes);
   if (eld)
   {
      for (int i = 0; i < 2; ++i)
      {
         FiniteElementCollection *eld_fec = new H1_FECollection(order, dim);
         ParFiniteElementSpace *eld_fes = new ParFiniteElementSpace(&pmesh, eld_fec, dim);
         pfes.Append(eld_fes);
      }
   }

   ScalarMatrixProductCoefficient m_cf_r(negeps0omeg2, eps_cf_r);
   ScalarMatrixProductCoefficient m_cf_i(negeps0omeg2, eps_cf_i);

   ParComplexBlockForm *a = new ParComplexBlockForm(pfes);
   // (1/μ₀ ∇×E, ∇ × F)
   a->AddDomainIntegrator(new CurlCurlIntegrator(muinv), nullptr, 0, 0);
   // - ω² ϵ₀ (ϵᵣ E, F)
   a->AddDomainIntegrator(new VectorFEMassIntegrator(m_cf_r),
                          new VectorFEMassIntegrator(m_cf_i), 0, 0);
   // if ELD
   ConstantCoefficient c1_cf(c1);
   ConstantCoefficient c2_cf(c2);
   ScalarMatrixProductCoefficient negc1Prbb_cf(-c1, P_cf_bb_r);
   ScalarMatrixProductCoefficient c2Prbb_cf(c2, P_cf_bb_r);
   // ConstantCoefficient eps0omeg2(eps0 * omega * omega);

   if (eld)
   {
      // - i ω²( (Jₕ⁽¹⁾ + Jₕ⁽²⁾), F)
      a->AddDomainIntegrator(nullptr, new TransposeIntegrator(new VectorFEMassIntegrator(negeps0omeg2)), 1, 0);
      a->AddDomainIntegrator(nullptr, new TransposeIntegrator(new VectorFEMassIntegrator(negeps0omeg2)), 2, 0);
      // (b ⊗ b ∇Jₕ⁽¹⁾ : ∇G)
      a->AddDomainIntegrator(new VectorDiffusionIntegrator(bb_cf),nullptr,1,1);
      // c₁ (Jₕ⁽¹⁾, G)
      a->AddDomainIntegrator(new VectorMassIntegrator(c1_cf), nullptr, 1, 1);
      // - c₁ (P(r) (b ⊗ b) E, G)
      a->AddDomainIntegrator(new VectorFEMassIntegrator(negc1Prbb_cf), nullptr, 0, 1);
      //  (b ⊗ b ∇Jₕ⁽²⁾ : ∇H)
      a->AddDomainIntegrator(new VectorDiffusionIntegrator(bb_cf), nullptr, 2, 2);
      // c₂ (Jₕ⁽²⁾, H)
      a->AddDomainIntegrator(new VectorMassIntegrator(c2_cf), nullptr, 2, 2);
      // c₂ (P(r) (b ⊗ b) E, H)
      a->AddDomainIntegrator(new VectorFEMassIntegrator(c2Prbb_cf), nullptr, 0, 2);
   }

   socketstream E_out_r;
   socketstream E_theta_out_r;
   socketstream E_theta_out_i;

   int npfes = pfes.Size();
   Array<int> offsets(npfes+1);
   offsets[0] = 0;
   for (int i = 0; i<npfes; i++)
   {
      offsets[i+1] = pfes[i]->GetVSize();
   }
   offsets.PartialSum();

   Vector x(2*offsets.Last());
   x = 0.;

   ParGridFunction E_gf_r(E_fes, x, 0); E_gf_r = 0.0;
   ParGridFunction E_gf_i(E_fes, x, offsets.Last()); E_gf_i = 0.0;
   ParGridFunction J_h1_r,J_h1_i,J_h2_r,J_h2_i;
   if (eld)
   {
      J_h1_r.MakeRef(pfes[1], x, offsets[1]);
      J_h1_i.MakeRef(pfes[1], x, offsets.Last() + offsets[1]);
      J_h2_r.MakeRef(pfes[2], x, offsets[2]);
      J_h2_i.MakeRef(pfes[2], x, offsets.Last() + offsets[2]); 
   }

   L2_FECollection L2fec(order, dim);
   ParFiniteElementSpace L2_fes(&pmesh, &L2fec);
   ParGridFunction E_par_r(&L2_fes);
   ParGridFunction E_par_i(&L2_fes);

   ParaViewDataCollection * paraview_dc = nullptr;

   std::string output_dir = "ParaView/FEM/" + GetTimestamp();


   if (paraview)
   {
      if (Mpi::Root()) { WriteParametersToFile(args, output_dir); }
      std::ostringstream paraview_file_name;
      std::string filename = GetFilename(mesh_file);
      paraview_file_name << filename
                         << "_par_ref_" << par_ref_levels
                         << "_order_" << order;
      paraview_dc = new ParaViewDataCollection(paraview_file_name.str(), &pmesh);
      paraview_dc->SetPrefixPath(output_dir);
      paraview_dc->SetLevelsOfDetail(order);
      paraview_dc->SetCycle(0);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetTime(0.0); // set the time
      paraview_dc->RegisterField("E_r",&E_gf_r);
      paraview_dc->RegisterField("E_i",&E_gf_i);
      paraview_dc->RegisterField("E_par_r",&E_par_r);
      paraview_dc->RegisterField("E_par_i",&E_par_i);
      if (eld)
      {
         paraview_dc->RegisterField("Jh_1_r",&J_h1_r);
         paraview_dc->RegisterField("Jh_1_i",&J_h1_i);
         paraview_dc->RegisterField("Jh_2_r",&J_h2_r);
         paraview_dc->RegisterField("Jh_2_i",&J_h2_i);
      }
   }

   Array<int> ess_tdof_list;
   Array<int> ess_tdof_listJ1;
   Array<int> ess_tdof_listJ2;
   Array<int> ess_bdr;
   Array<int> one_r_bdr;
   Array<int> one_i_bdr;
   Array<int> negone_r_bdr;
   Array<int> negone_i_bdr;

   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
      one_r_bdr.SetSize(pmesh.bdr_attributes.Max());
      one_i_bdr.SetSize(pmesh.bdr_attributes.Max());
      negone_r_bdr.SetSize(pmesh.bdr_attributes.Max());
      negone_i_bdr.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      // remove internal boundaries
      for (int i = 0; i<int_bdr_attr.Size(); i++)
      {
         ess_bdr[int_bdr_attr[i]-1] = 0;
      }

      E_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      if (eld)
      {
         pfes[1]->GetEssentialTrueDofs(ess_bdr, ess_tdof_listJ1);
         pfes[2]->GetEssentialTrueDofs(ess_bdr, ess_tdof_listJ2);
         for (int i = 0; i < ess_tdof_listJ1.Size(); i++)
         {
            ess_tdof_listJ1[i] += pfes[0]->GetTrueVSize();
            ess_tdof_listJ2[i] += pfes[0]->GetTrueVSize() + pfes[1]->GetTrueVSize();
         }
      }
      one_r_bdr = 0;  one_i_bdr = 0;
      negone_r_bdr = 0;  negone_i_bdr = 0;

      // attr = 30,2 (real)
      one_r_bdr[30-1] = 1;  one_r_bdr[2-1] = 1;
      // attr = 26,6 (imag)
      one_i_bdr[26-1] = 1;  one_i_bdr[6-1] = 1;
      // attr = 22,10 (real)
      negone_r_bdr[22-1] = 1; negone_r_bdr[10-1] = 1;
      // attr = 18,14 (imag)
      negone_i_bdr[18-1] = 1; negone_i_bdr[14-1] = 1;
   }

   if (eld)
   {
      ess_tdof_list.Append(ess_tdof_listJ1);
      ess_tdof_list.Append(ess_tdof_listJ2);
   }

   Vector zero(dim); zero = 0.0;
   Vector one_x(dim); one_x = 0.0; one_x(0) = 1.0;
   Vector negone_x(dim); negone_x = 0.0; negone_x(0) = -1.0;
   VectorConstantCoefficient zero_vcf(zero);
   VectorConstantCoefficient one_x_cf(one_x);
   VectorConstantCoefficient negone_x_cf(negone_x);

   E_gf_r.ProjectBdrCoefficientTangent(one_x_cf, one_r_bdr);
   E_gf_r.ProjectBdrCoefficientTangent(negone_x_cf, negone_r_bdr);
   E_gf_i.ProjectBdrCoefficientTangent(one_x_cf, one_i_bdr);
   E_gf_i.ProjectBdrCoefficientTangent(negone_x_cf, negone_i_bdr);

   a->Assemble();

   OperatorPtr Ah;
   Vector B, X;

   Vector b(x.Size()); b = 0.0;
   a->FormLinearSystem(ess_tdof_list, x, b, Ah, X, B);
   
   ComplexOperator * Ahc = Ah.As<ComplexOperator>();

   BlockOperator * BlockA_r = dynamic_cast<BlockOperator *>(&Ahc->real());
   BlockOperator * BlockA_i = dynamic_cast<BlockOperator *>(&Ahc->imag());

   int nblocks = BlockA_r->NumRowBlocks();
   Array2D<HypreParMatrix*> A_r_matrices(nblocks, nblocks);
   Array2D<HypreParMatrix*> A_i_matrices(nblocks, nblocks);
   for (int i = 0; i < nblocks; i++)
   {
      for (int j = 0; j < nblocks; j++)
      {
         A_r_matrices(i,j) = dynamic_cast<HypreParMatrix*>(&BlockA_r->GetBlock(i,j));
         A_i_matrices(i,j) = dynamic_cast<HypreParMatrix*>(&BlockA_i->GetBlock(i,j));
      }
   }
   HypreParMatrix * Ahr = HypreParMatrixFromBlocks(A_r_matrices);
   HypreParMatrix * Ahi = HypreParMatrixFromBlocks(A_i_matrices);

   ComplexHypreParMatrix * Ahc_hypre =
      new ComplexHypreParMatrix(Ahr, Ahi,false, false);

   if (Mpi::Root())
   {
      mfem::out << "Assembly finished successfully." << endl;
   }

#ifdef MFEM_USE_MUMPS
   HypreParMatrix *A = Ahc_hypre->GetSystemMatrix();
   auto solver = new MUMPSSolver(MPI_COMM_WORLD);
   solver->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
   solver->SetPrintLevel(1);
   solver->SetOperator(*A);
   solver->Mult(B,X);
   delete A;
   delete solver;
#else
   MFEM_ABORT("MFEM compiled without mumps");
#endif

   a->RecoverFEMSolution(X, x);


   E_gf_r.MakeRef(E_fes, x, 0); 
   E_gf_i.MakeRef(E_fes, x, offsets.Last());
   if (eld)
   {
      J_h1_r.MakeRef(pfes[1], x, offsets[1]);
      J_h1_i.MakeRef(pfes[1], x, offsets.Last() + offsets[1]);
      J_h2_r.MakeRef(pfes[2], x, offsets[2]);
      J_h2_i.MakeRef(pfes[2], x, offsets.Last() + offsets[2]); 
   }
   ParallelECoefficient par_e_r(&E_gf_r);
   ParallelECoefficient par_e_i(&E_gf_i);
   E_par_r.ProjectCoefficient(par_e_r);
   E_par_i.ProjectCoefficient(par_e_i);

   if (visualization)
   {
      const char * keys = nullptr;
      char vishost[] = "localhost";
      int  visport   = 19916;
      common::VisualizeField(E_out_r,vishost, visport, E_gf_r,
                             "Numerical Electric field (real part)", 0, 0, 500, 500, keys);
   }

   if (paraview)
   {
      paraview_dc->SetCycle(0);
      paraview_dc->SetTime((real_t)0);
      paraview_dc->Save();
      delete paraview_dc;
   }


   delete a;
   delete E_fes;
   delete fec;

   return 0;

}
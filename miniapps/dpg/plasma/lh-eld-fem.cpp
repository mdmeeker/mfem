//                   MFEM Ultraweak DPG Maxwell parallel example
//
// Strong formulation:
//      ∇×(1/μ₀ ∇×E) - ω² ϵ₀ ϵᵣ E - i ω²(Jₕ⁽¹⁾ + Jₕ⁽²⁾) = 0,   in Ω
//                    - Δ∥ Jₕ⁽¹⁾ + c₁ Jₕ⁽¹⁾ - c₁ P(r) E∥ = 0,   in Ω     
//                    - Δ∥ Jₕ⁽²⁾ + c₂ Jₕ⁽²⁾ + c₂ P(r) E∥ = 0,   in Ω 
//                                                  E×n = E₀,  on ∂Ω
// weak formulation:
//   Find E ∈ H(curl,Ω), Jₕ⁽¹⁾ ∈ H¹(Ω), Jₕ⁽²⁾ ∈ H¹(Ω) such that
//   (1/μ₀ ∇×E, ∇ × F) - ω² ϵ₀ (ϵᵣ E, F) - i ω²( (Jₕ⁽¹⁾ + Jₕ⁽²⁾), F) = 0,   ∀  F ∈ H(curl,Ω)
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
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
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

   ScalarMatrixProductCoefficient eps0eps_r(eps0, eps_cf_r);
   ScalarMatrixProductCoefficient eps0eps_i(eps0, eps_cf_i);

   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *E_fes = new ParFiniteElementSpace(&pmesh, fec);

   // Bilinear form coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient muinv(1./mu);

   ScalarMatrixProductCoefficient m_cf_r(-omega*omega, eps0eps_r);
   ScalarMatrixProductCoefficient m_cf_i(-omega*omega, eps0eps_i);


   Array<ParFiniteElementSpace *> pfes;
   pfes.Append(E_fes);

   ParComplexLinearForm *b = new ParComplexLinearForm(E_fes);
   b->Vector::operator=(0.0);


   ParComplexBlockForm *a = new ParComplexBlockForm(pfes);
   a->AddDomainIntegrator(new CurlCurlIntegrator(muinv), nullptr, 0, 0);
   a->AddDomainIntegrator(new VectorFEMassIntegrator(m_cf_r),
                          new VectorFEMassIntegrator(m_cf_i), 0, 0);

   socketstream E_out_r;
   socketstream E_theta_out_r;
   socketstream E_theta_out_i;

   ParComplexGridFunction E_gf(E_fes);
   E_gf.real() = 0.0;
   E_gf.imag() = 0.0;

   L2_FECollection L2fec(order, dim);
   ParFiniteElementSpace L2_fes(&pmesh, &L2fec);

   ParGridFunction E_theta_r(&L2_fes);
   ParGridFunction E_theta_i(&L2_fes);
   ParGridFunction E_theta(&L2_fes);
   E_theta = 0.0;

   ParGridFunction E_par_r(&L2_fes);
   ParGridFunction E_par_i(&L2_fes);

   ParaViewDataCollection * paraview_dc = nullptr;
   // ParaViewDataCollection * paraview_tdc = nullptr;

   std::string output_dir = "ParaView/FEM/" + GetTimestamp();
   if (Mpi::Root())
   {
      WriteParametersToFile(args, output_dir);
   }

   if (paraview)
   {
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
      paraview_dc->RegisterField("E_r",&E_gf.real());
      paraview_dc->RegisterField("E_i",&E_gf.imag());
      paraview_dc->RegisterField("E_theta_r",&E_theta_r);
      paraview_dc->RegisterField("E_theta_i",&E_theta_i);
      paraview_dc->RegisterField("E_par_r",&E_par_r);
      paraview_dc->RegisterField("E_par_i",&E_par_i);
   }

   Array<int> ess_tdof_list;
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

   Vector zero(dim); zero = 0.0;
   Vector one_x(dim); one_x = 0.0; one_x(0) = 1.0;
   Vector negone_x(dim); negone_x = 0.0; negone_x(0) = -1.0;
   VectorConstantCoefficient zero_vcf(zero);
   VectorConstantCoefficient one_x_cf(one_x);
   VectorConstantCoefficient negone_x_cf(negone_x);

   E_gf.ProjectBdrCoefficientTangent(one_x_cf,zero_vcf, one_r_bdr);
   E_gf.ProjectBdrCoefficientTangent(negone_x_cf,zero_vcf, negone_r_bdr);
   E_gf.ProjectBdrCoefficientTangent(zero_vcf,one_x_cf, one_i_bdr);
   E_gf.ProjectBdrCoefficientTangent(zero_vcf,negone_x_cf, negone_i_bdr);

   b->Assemble();
   a->Assemble();

   OperatorPtr Ah;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, E_gf, *b, Ah, X, B);
   
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

#ifdef MFEM_USE_MUMPS
   HypreParMatrix *A = Ahc_hypre->GetSystemMatrix();
   // auto cpardiso = new CPardisoSolver(A->GetComm());
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

   a->RecoverFEMSolution(X, E_gf);

   AzimuthalECoefficient az_e_r(&E_gf.real());
   AzimuthalECoefficient az_e_i(&E_gf.imag());
   E_theta_r.ProjectCoefficient(az_e_r);
   E_theta_i.ProjectCoefficient(az_e_i);

   ParallelECoefficient par_e_r(&E_gf.real());
   ParallelECoefficient par_e_i(&E_gf.imag());
   E_par_r.ProjectCoefficient(par_e_r);
   E_par_i.ProjectCoefficient(par_e_i);

   if (visualization)
   {
      const char * keys = nullptr;
      char vishost[] = "localhost";
      int  visport   = 19916;
      common::VisualizeField(E_out_r,vishost, visport, E_gf.real(),
                             "Numerical Electric field (real part)", 0, 0, 500, 500, keys);
      common::VisualizeField(E_theta_out_r,vishost, visport, E_theta_r,
                             "Numerical Electric field (azimuthal)", 0, 0, 500, 500, keys);
   }

   if (paraview)
   {
      paraview_dc->SetCycle(0);
      paraview_dc->SetTime((real_t)0);
      paraview_dc->Save();
      delete paraview_dc;

      // int num_frames = 32;
      // for (int i = 0; i<num_frames; i++)
      // {
      //    real_t t = (real_t)(i % num_frames) / num_frames;
      //    add(cos(real_t(2.0*M_PI)*t), E_theta_r,
      //        sin(real_t(2.0*M_PI)*t), E_theta_i, E_theta);
      //    paraview_tdc->SetCycle(i);
      //    paraview_tdc->SetTime(t);
      //    paraview_tdc->Save();
      // }
      // delete paraview_tdc;
   }


   delete a;
   delete b;
   delete E_fes;
   delete fec;

   return 0;

}
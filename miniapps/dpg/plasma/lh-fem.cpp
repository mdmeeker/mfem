//                   MFEM Ultraweak DPG Maxwell parallel example
//
// the "ultraweak" (UW) DPG formulation for the Maxwell problem

//      ∇×(1/μ ∇×E) - ω² ϵ E = Ĵ ,   in Ω
//                       E×n = E₀ , on ∂Ω
#include "mfem.hpp"
#include "../util/pcomplexweakform.hpp"
#include "../util/utils.hpp"
#include "../util/maxwell_utils.hpp"
#include "../../common/mfem-common.hpp"
#include <fstream>
#include <iostream>
#include <cstring>
#include <filesystem>

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // fine mesh (trianles)
   // default mesh
   const char *mesh_file = "data/mesh-tri34K.mesh";
   // coarse mesh (triangles)
   // const char *mesh_file = "data/mesh-tri11K.mesh";
   // coarse mesh (quadrilaterals)
   // const char *mesh_file = "data/mesh-quad5K.mesh";

   // epsilon tensor
   const char * eps_r_file = nullptr;
   const char * eps_i_file = nullptr;

   int order = 2;
   int par_ref_levels = 0;
   bool visualization = false;
   real_t rnum=1.5e9;
   real_t mu = 1.257e-6;
   real_t eps0 = 8.8541878128e-12;
   // real_t rnum=1.5;
   // real_t mu = 1.257;
   // real_t eps0 = 8.8541878128;

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
   
   real_t delta = 0.01; // dielectric constant
   real_t a0 = -1.0;    // dielectric constant for the first component
   real_t a1 = 5.0;    // dielectric constant for the second component
   MatrixArrayCoefficient eps_r_cf(dim);
   MatrixArrayCoefficient eps_i_cf(dim);
   ConstantCoefficient one_cf(1.0);
   ConstantCoefficient zero_cf(0.0);
   Array<Coefficient*> coeff_array_one(nattr);
   Array<Coefficient*> coeff_array_zero(nattr);
   
   for (int i = 0; i < nattr-1; ++i)
   {
      coeff_array_one[i] = &one_cf;
      coeff_array_zero[i] = &zero_cf;
   }
   for (int i = 0; i < dim; ++i)
   {
      for (int j = 0; j < dim; ++j)
      {
         auto *coeff_r = new DielectricTensorComponentCoefficient(
               delta, a0, a1, i, j, false);
         PWCoefficient * pw_coeff_r = nullptr;   
         if (i == j)
         {
            coeff_array_one[nattr-1] = coeff_r;
            pw_coeff_r = new PWCoefficient(attr,coeff_array_one);   
         }      
         else
         {
            coeff_array_zero[nattr-1] = coeff_r;
            pw_coeff_r = new PWCoefficient(attr,coeff_array_zero);
         }

         eps_r_cf.Set(i, j, pw_coeff_r);

         auto *coeff_i = new DielectricTensorComponentCoefficient(
               delta, a0, a1, i, j, true);

         coeff_array_zero[nattr-1] = coeff_i;      

         auto *pw_coeff_i = new PWCoefficient(attr,coeff_array_zero);

         eps_i_cf.Set(i, j, pw_coeff_i);
      }
   }  

   // VisualizeMatrixArrayCoefficient(eps_r_cf, &pmesh, order, paraview, "real_eps"); 
   // VisualizeMatrixArrayCoefficient(eps_i_cf, &pmesh, order, paraview, "imag_eps");
   // cin.get();

   // eps_r_file = "data/eps-tri34K_r.gf";
   // eps_i_file = "data/eps-tri34K_i.gf";

   // EpsilonMatrixCoefficient eps_r_cf(eps_r_file,&mesh,&pmesh, eps0);
   // EpsilonMatrixCoefficient eps_i_cf(eps_i_file,&mesh,&pmesh, eps0);


   // VisualizeMatrixArrayCoefficient(eps_r_cf, &pmesh, order, paraview, "file_real_eps"); 
   // VisualizeMatrixArrayCoefficient(eps_i_cf, &pmesh, order, paraview, "file_imag_eps");
   // cin.get();

   for (int i = 0; i<par_ref_levels; i++)
   {
      pmesh.UniformRefinement();
      // eps_r_cf.Update();
      // eps_i_cf.Update();
   }

   ScalarMatrixProductCoefficient eps0eps_r(eps0, eps_r_cf);
   ScalarMatrixProductCoefficient eps0eps_i(eps0, eps_i_cf);

   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *E_fes = new ParFiniteElementSpace(&pmesh, fec);

   // Bilinear form coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient muinv(1./mu);

   ScalarMatrixProductCoefficient m_cf_r(-omega*omega, eps0eps_r);
   ScalarMatrixProductCoefficient m_cf_i(-omega*omega, eps0eps_i);

   ParComplexLinearForm *b = new ParComplexLinearForm(E_fes);
   b->Vector::operator=(0.0);

   ParSesquilinearForm *a = new ParSesquilinearForm(E_fes);
   a->AddDomainIntegrator(new CurlCurlIntegrator(muinv),nullptr);
   a->AddDomainIntegrator(new VectorFEMassIntegrator(m_cf_r),
                          new VectorFEMassIntegrator(m_cf_i));

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

      // paraview_tdc = new ParaViewDataCollection(mesh_file, &pmesh);
      // paraview_tdc->SetPrefixPath("ParaViewFEM2D/TimeHarmonic");
      // paraview_tdc->SetLevelsOfDetail(order);
      // paraview_tdc->SetCycle(0);
      // paraview_tdc->SetDataFormat(VTKFormat::BINARY);
      // paraview_tdc->SetHighOrderOutput(true);
      // paraview_tdc->SetTime(0.0); // set the time
      // paraview_tdc->RegisterField("E_theta_t",&E_theta);
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

#ifdef MFEM_USE_MUMPS
   HypreParMatrix *A = Ah.As<ComplexHypreParMatrix>()->GetSystemMatrix();
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

   a->RecoverFEMSolution(X, *b, E_gf);

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
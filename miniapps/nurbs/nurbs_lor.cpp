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
   // ----- Test GetInterpolationMatrix -----
   // Build a patch from scratch
   const int tdim = 2;
   const int pdim = 2;
   Array<const KnotVector*> kvs(tdim);
   kvs[0] = new KnotVector(2, Vector({0.0, 1.0}));
   kvs[1] = new KnotVector(1, Vector({0.0, 1.0}));
   Vector control_points(
   {
      0.0, 0.0, 1.0,
      1.0, 0.0, 1.0,
      2.0, 0.0, 1.0,
      0.0, 1.0, 1.0,
      1.0, 1.0, 1.0,
      2.0, 1.0, 1.0,
   });
   NURBSPatch patch(kvs, pdim, control_points.GetData());

   // Define new knots
   Array<Vector *> uknots;
   uknots.SetSize(tdim);
   uknots[0] = new Vector({0.4, 0.80});
   uknots[1] = new Vector({0.3});

   // ----- Get the interpolation matrix R -----
   /** R should be a 2x3 matrix with values:
       0.252  0.336  0.112  0.108  0.144  0.048
       0.028  0.224  0.448  0.012  0.096  0.192
    */
   SparseMatrix R = patch.GetInterpolationMatrix(uknots);

   // Print as dense matrix
   cout << "R = " << endl;
   R.ToDenseMatrix()->PrintMatlab(cout);


   // Free memory
   for (int i = 0; i < tdim; i++)
   {
      delete kvs[i];
      delete uknots[i];
   }

   return 0;
}

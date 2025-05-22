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
   KnotVector kv(2, Vector({0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0}));
   kv.Print(cout);

   const real_t ui = 3.5;
   Vector shape(kv.GetOrder()+1);
   int kidx = kv.CalcShape(shape, ui);
   cout << "kidx = " << kidx << endl;
   shape.Print(cout);


   // test calcshapes
   Vector newknots({0.5, 1.5, 2.0, 2.0, 2.5, 3.5, 4.5});
   std::vector<KnotVector::ShapeValues> shapes = kv.CalcShapes(newknots);
   cout << "knot 0 = " << shapes[0].u << endl;
   shapes[0].shape.Print();

   return 0;
}

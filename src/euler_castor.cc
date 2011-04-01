#include "castor.h"
#include "iostream"
#include "stdint.h"
#include <boost/function.hpp>


namespace cs = castor;

int32_t prob1() {
  cs::lref<int32_t> x; 
  int32_t num = 999;
  int32_t answer = 0;  
  cs::relation multiples = 
        (cs::range<int32_t>(x, 1, num) 
         && (cs::predicate(x % 5 == 0) 
        ^ cs::predicate(x % 3 == 0))
      );
  while (multiples()) 
     answer += *x;
  return answer;
}

// boost::function<cs::lref<int> (cs::lref<int> &x, cs::lref<int> &y, cs::lref<int> &z)> func; 
// cs::lref<int> incf(cs::lref<int> &x,cs::lref<int> &y,cs::lref<int> &z) {
//     z = x * y;
//     x = y;
//     y = z;
//     return z;
//   }
//    
int64_t prob2() { 
  int32_t max = 4000000;
//   cs::lref<int> x ;
//   cs::lref<int> y ;
//   cs::lref<int> z ;
  //(z=x+y);
//   (y=z);
//   (x=y);


//   func = &incf;
  int count = 3;
  cs::relation answer(cs::lref<int> x, cs::lref<int> y) { 
                            cs::range<int>(count, 1, max)
                         && cs::predicate(z < max)
                         && cs::predicate(z % 2 == 0)
                         && cs::recurse answer(x , y)
}
//                          && cs::eq_f(y, z *1)

  while(answer())
    std::cout << *z << std::endl;
  return *z;
  }


int prob9() {
  cs::lref<int> x, y, z; 
  int num = 1000;
  cs::relation triplets = cs::range<int>(x, 1, num)
                          && cs::range<int>(y, 1, num)
                          && cs::range<int>(z, 1, num)
                          && cs::predicate(x + y + z == num)
                          && (cs::predicate(x*x + y*y == z*z) 
                          || cs::predicate(z*z + y*y == x * x) );
   if (triplets()) 
     return *x * *z * *y;
   else
     return 0;
}


// ###### MAIN ######
int main() {
//   std::cout << "Prob 1: " << prob1() << std::endl;
//   std::cout << "Prob 2: " << prob2() << std::endl;
  std::cout << "Prob 2: " << prob2() << std::endl;
  //   std::cout << "Prob 9: " << prob9() << std::endl;
  

  return 0;
}

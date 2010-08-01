#include <iostream>
#include "castor.h"

castor::relation child(castor::lref<std::string> c,
                       castor::lref<std::string> p) {
  return eq(c, "David") && eq(p, "Morag")
  ||     eq(c, "David") && eq(p, "William");
}

castor::relation gender(castor::lref<std::string> p,
                        castor::lref<std::string> g) {
  return eq(p, "William") && eq(g, "Male")
  ||     eq(p, "David") && eq(g, "Male")
  ||     eq(p, "David")   && eq(g,"SuperMale");
}

castor::relation father(castor::lref<std::string> f,
                        castor::lref<std::string> c){
  return gender(f, "Male") && child(c, f);
}

int main(int argc, char **argv) {
  castor::relation IsDavidMale = gender("David", "Male");
  castor::relation IsFather = father("William", "David");
// Simple test for truth 
if (IsDavidMale()) {
  std::cout << "David is, in fact, male, of course " << std::endl;
} else {
  std::cout << "David is not male, there is an error !! " << std::endl;
}
// And again 
if (IsFather()) {
  std::cout << "William is David's Father" << std::endl;
} else {
  std::cout << "william is not David's father" << std::endl;
}
// empty logical reference here - filled in when we ask for it
// interestingly this overwrites the pointer target continually
// so it's way faster than iterating
castor::lref<std::string> g;
castor::relation DavidsGender = gender("David", g);
while (DavidsGender())
  std::cout << "David is " << *g << std::endl;


}

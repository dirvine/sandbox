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
  ||     eq(p, "David") && eq(g, "Male");
}

castor::relation father(castor::lref<std::string> f,
                        castor::lref<std::string> c){
  return gender(f, "Male") && child(c, f);
}

int main(int argc, char **argv) {
  castor::relation IsSamMail = gender("David", "Male");
  castor::relation IsFather = father("William", "David");

if (IsSamMail()) {
  std::cout << "David is, in fact, male, of course " << std::endl;
} else {
  std::cout << "David is not male" << std::endl;
}

if (IsFather()) {
  std::cout << "William is David's Father" << std::endl;
} else {
  std::cout << "william is not David's father" << std::endl;
}

castor::lref<std::string> g;
castor::relation SamsGender = gender("Sam", g);

}

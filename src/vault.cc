// #include <iostream>
// #include <algorithm>
// #include <boost/cstdint.hpp>
// #include <boost/function.hpp>
// #include "castor.h"
// /*
// *****facts *****
// /My stuff
// my_node_id
//
// // Node Account stuff
// distance_from_me (xor)
// space_used
// space_available
// rank ??
// score // highest score gets chunk or receive chunk from
//
// // Stats
// rtt // Use as a crude distance measure for now
// send_bandwidth
// receive_bandwidth
// last_seen // if very recent then act - may be harmful to network
//           // should be first seen in the new kad routing table - longest value
//           // being best nodes in a session measurement.
//           // using space used etc. we cna calculate a dynamic rank
//           // ourselves, mix with recorded rank for getting better/worse
//           // (debatable worse)
// direct_connection
// requires_rendezvous
// tcp_only
// udp_capable
//
//
// // RPC stuff
// node_in_question
// value_in_question
//
// **** Rules *****
//
// ******************
// //Golden rules
// 1: Network [ protect at all costs !!]
// 2: Data [ protect data over self ]
// 3: Self [ protect self last, i.e. we are expendable for the greater good ]
//
// // Routing table wide Alert for very bad nodes !!! save network, all attack !!
// // With LP we can go traverse a list 100 times faster than stl so go through all
// // routing table every time unless we can improve algorithm (which we can)
// */
// // In SentiNet catch add/remove Kad nodes
//
// // For kademlia
// castor::relation rank(castor::lref<std::string> node,
//                     castor::lref<boost::uint32_t> rank);
// namespace cs = castor;
//
// // For SentiNet
//
// castor::relation first_seen(castor::lref<std::string> node,
//                             castor::lref<boost::uint32_t> time);
// castor::relation already_used(castor::lref<std::string> node,
//                               castor::lref<boost::uint32_t> stored);
// castor::relation send_bandwidth(castor::lref<std::string> node,
//                                 castor::lref<boost::uint32_t> send_bw);
// castor::relation receive_bandwidth(castor::lref<std::string> node,
//                                   castor::lref<boost::uint32_t> receive_bw);
// castor::relation rtt(cs::lref<std::string> node,
//                      cs::lref<boost::uint32_t> r_t_t);
// struct Ant {
//   Ant(){}
//   ~Ant(){}
// public:
//
//
// };
//
// enum AntType{kSoldier, kForger, kCollector, kCleaner, kSleeper};
// enum AntFunction{kProtect, kFindFood, kCarryFood, kClean, kSleep};
//
// cs::relation DefaultFunction(cs::lref<AntType> t, cs::lref<AntFunction> f) {
//   return cs::eq(t, kSoldier) && cs::eq(f, kProtect)
//   ||     cs::eq(t, kForger) && cs::eq(f, kFindFood)
//   ||     cs::eq(t, kCollector) && cs::eq(f, kCarryFood)
//   ||     cs::eq(t,kCleaner) && cs::eq(f, kClean)
//   ||     cs::eq(t,kSleeper) && cs::eq(f, kSleep);
// }
//
// void pythagorus() {
//     // Print all Pythagoras triplets less
//     // than x
//     cs::lref<int> x,y,z;
//     int max=50;
//     cs::relation pythTriplets = cs::range<int>(x,1,max) && cs::range<int>(y,1,max)
//                   && cs::range<int>(z,1,max) && (cs::predicate(x*x+y*y==z*z)
//                   || cs::predicate(z*z+y*y==x*x));
//     while(pythTriplets())
//         std::cout << *x << "," << *y << ","<< *z << "\n";
// }
//
//
// cs::Disjunctions TestFunc() {
//
// }
//
// int CreateAnt() {
//   int t = 0;
//   return t;
// }
//
// int main(int argc, char **argv) {
//  pythagorus();
//   return 0;
// }

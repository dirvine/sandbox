#include <iostream>
#include <algorithm>
#include <boost/cstdint.hpp>
#include "castor.h"
/*
*****facts *****
/My stuff
my_node_id

// Node Account stuff
distance_from_me (xor)
space_used
space_available
rank ??
score // highest score gets chunk or receive chunk from

// Stats
rtt // Use as a crude distance measure for now
send_bandwidth
receive_bandwidth
last_seen // if very recent then act - may be harmful to network
          // should be first seen in the new kad routing table - longest value
          // being best nodes in a session measurement.
          // using space used etc. we cna calculate a dynamic rank
          // ourselves, mix with recorded rank for getting better/worse
          // (debatable worse)
direct_connection
requires_rendezvous
tcp_only
udp_capable


// RPC stuff
node_in_question
value_in_question

**** Rules *****




******************
//Golden rules
1: Network [ protect at all costs !!]
2: Data [ protect data over self ]
3: Self [ protect self last, i.e. we are expendable for the greater good ]

// Routing table wide Alert for very bad nodes !!! save network, all attack !!
// With LP we can go traverse a list 100 times faster than stl so go through all
// routing table every time unless we can improve algorithm (which we can)
*/
// In SentiNet catch add/remove Kad nodes

// For kademlia
castor::relation rank(castor::lref<std::string> node,
                    castor::lref<boost::uint32_t> rank);
                    



// For SentiNet
                      
castor::relation first_seen(castor::lref<std::string> node,
                            castor::lref<boost::uint32_t> time);

castor::relation already_used(castor::lref<std::string> node,
                              castor::lref<boost::uint32_t> stored);

castor::relation send_bandwidth(castor::lref<std::string> node,
                                castor::lref<boost::uint32_t> send_bw);
                                
castor::relation receive_bandwidth(castor::lref<std::string> node,
                      castor::lref<boost::uint32_t> receive_bw);
                    
castor::relation rtt(castor::lref<std::string> node,
                      castor::lref<boost::uint32_t> r_t_t);


                      

int main(int argc, char **argv) {

}
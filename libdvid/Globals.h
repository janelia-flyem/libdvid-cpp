/*!
 * This file defines some global variables used
 * throughout the library.
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/

#ifndef DVIDGLOBALS_H
#define DVIDGLOBALS_H

#include <boost/cstdint.hpp>
#include <climits>

namespace libdvid {

typedef boost::uint8_t uint8;
typedef boost::uint64_t uint64;

//! By default everything in DVID has 32x32x32 blocks
const int DEFBLOCKSIZE = 32;

//! Gives the limit for how many vertice can be operated on in one call
const int TransactionLimit = 1000;

}

#endif

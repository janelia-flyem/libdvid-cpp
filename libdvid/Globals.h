/*!
 * This file defines some global variables used
 * throughout the library.
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/

#ifndef DVIDGLOBALS_H
#define DVIDGLOBALS_H

#include <boost/static_assert.hpp>
#include <climits>

namespace libdvid {

//! Asssume long long are 64 bits
//! TODO: make more robust for more build environments
typedef unsigned long long uint64;

//! Ensure uint64 is 8 bytes
BOOST_STATIC_ASSERT(sizeof(uint64) == 8);

//! By default everything in DVID has 32x32x32 blocks
const int DEFBLOCKSIZE = 32;

//! Grayscale type is 1 byte
typedef unsigned char uint8;

//! Gives the limit for how many vertice can be operated on in one call
const int TransactionLimit = 1000;

}

#endif

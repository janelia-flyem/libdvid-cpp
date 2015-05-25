/*!
 * This file provides algorithms to quickly fetch several smaller
 * pieces of data from DVID in a more efficient way.  To this end,
 * it supports the ability to fetch data in parallel.  If threading
 * is enabled, the DVID backend should ideally be a distributed one.
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/
#ifndef THREADEDFETCH 
#define THREADEDFETCH

// TODO: implement a copy constructor for DVIDNodeService

#include "DVIDNodeService.h"

namespace libdvid {

/*!
 * Fetches all the grayscale blocks that intersect the body id in the specified
 * label volume.  If threading is enabled, multiple requests will be done
 * simultaneously.  This call tries to minimize the number of http requests
 * by asking for contiguous chunks that include the necessary blocks.
 * \param service name of dvid node service
 * \param labelvol_name name of label volume with body id
 * \param grayscale_name name of grayscale data instance
 * \param use_blocks if true uses block interface instead of raw ND
 * \param num_threads number of threads used in the fetch.
 * \param request_efficiency how requests are packaged (0: 1 at a time, 1: X contig)
 * \return grayscale blocks in X,Y,Z order of the body blocks
*/
GrayscaleBlocks get_body_blocks(DVIDNodeService& service,
        std::string labelvol_name, std::string grayscale_name, uint64 bodyid,
        bool use_blocks = false, int num_threads = 1,
        int request_efficiency = 1);


}

#endif

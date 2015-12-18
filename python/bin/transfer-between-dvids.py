#!/usr/bin/env python
"""
Transfer voxels data from one DVID server to another.
To avoid errors due to overly large requests, the data 
will be requested in blocks according to the --request-shape-xyz parameter.

NOTE: This script uses a few utility functions from ilastik's 'lazyflow' package,
      so you need ilastik installed.

Example usage:

    python transfer-between-dvids.py \
                --request-shape-xyz='(512,512,512)' \
                emdata2:7000 \
                fe3791 \
                grayscale \
                http://bergs-ws1.janelia.priv:8000 \
                9ace850166ee44f783aa9990f4edb926 \
                grayscale \
                '(10016, 5024, 10016)' \
                '(11040, 6048, 11040)' \
    ##

"""
import argparse
import logging
import numpy as np
import libdvid
from libdvid.voxels import VoxelsAccessor, DVID_BLOCK_WIDTH
from lazyflow.roi import getIntersectingBlocks, getBlockBounds, roiFromShape

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request-shape-xyz', default='(512, 512, 512)')
    parser.add_argument('source_server')
    parser.add_argument('source_uuid')
    parser.add_argument('source_instance')
    parser.add_argument('destination_server')
    parser.add_argument('destination_uuid')
    parser.add_argument('destination_instance')
    parser.add_argument('region_start_xyz')
    parser.add_argument('region_stop_xyz')
    args = parser.parse_args()

    request_shape = eval(args.request_shape_xyz)
    request_shape = np.array(request_shape)
    if any(request_shape.flat[:] % DVID_BLOCK_WIDTH):
        sys.exit("The request-shape dimensions must be multiples of {}".format(DVID_BLOCK_WIDTH))

    region_start = eval(args.region_start_xyz)
    region_stop = eval(args.region_stop_xyz)

    transfer(args.source_server,
             args.source_uuid,
             args.source_instance,
             args.destination_server,
             args.destination_uuid,
             args.destination_instance,
             (region_start, region_stop),
             request_shape )

def transfer(source_server, source_uuid, source_instance,
             destination_server, destination_uuid, destination_instance,
             region_bounds_xyz, request_shape_xyz):
    """
    Transfer voxels data from one DVID server to another.
    
    region_bounds_xyz: The region of data to transfer, as a tuple: (start,stop).
                       Must be aligned to DVID block boundaries.
                       For example: ((0,0,0), (1024, 1024, 512))

    request_shape_xyz: The data will be transfered one block at a time.
                       Each requeted block will have this shape 
                       (except that blocks near the edge of the volume will be smaller).
    
    """
    source_vol = VoxelsAccessor(source_server, source_uuid, source_instance)
    destination_vol = VoxelsAccessor(destination_server, destination_uuid, destination_instance)

    region_bounds_xyz = np.asarray(region_bounds_xyz)
    region_shape = np.subtract(*region_bounds_xyz[::-1])
    offset_block_starts = getIntersectingBlocks(request_shape_xyz, roiFromShape(region_shape))

    for offset_block_start in offset_block_starts:
        offset_block_bounds = getBlockBounds(region_shape, request_shape_xyz, offset_block_start)
        block_bounds = offset_block_bounds + region_bounds_xyz[0]
        block_bounds = np.concatenate( ([[0],[1]], block_bounds), axis=1)

        logger.debug("Requesting block: {}".format(block_bounds[:,1:]) )
        block = source_vol.get_ndarray( *block_bounds )

        logger.debug("Writing block: {}".format(block_bounds[:,1:]) )
        destination_vol.post_ndarray(*block_bounds, new_data=block)

    logger.debug("DONE.")

if __name__ == "__main__":
    import sys
    logger.addHandler( logging.StreamHandler(sys.stdout) )
    logger.setLevel(logging.DEBUG)

    main()

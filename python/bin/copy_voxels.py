#!/usr/bin/env python
"""
Copy voxels data within a ROI from one DVID instance to another
instance, possibly on a different server.

Example usage:

    # Source must be provided as a URL.  Destination and ROI can optionally be URLs,
    # or just instance names (which are assumed to live on the source server).

    python copy_voxels.py \
                --roi=my-region
                emdata2:7000/api/node/fe3791/grayscale \
                grayscale_2

    python copy_voxels.py \
                --roi=emdata1:7000/api/node/fe3791/compartment-1
                emdata2:8000/api/node/deadbeef/labels \
                bergs-ws1.janelia.priv:8000/9ace850166ee44f783aa9990f4edb926/labels
    ##

Note: The pixels within the ROI will be transferred in large blocks
      (using the roi partition function from DVID), and ALL pixels in
      those blocks will be overwritten, even those OUTSIDE of the roi!!

TODO: Here we import a hard-coded DVID_BLOCK_WIDTH from libdvid.voxels instead of
      reading it from the DVID Server.  That will be bad when we start using DVID
      servers with 64-px blocks!
"""
import re
import argparse
import collections
import logging
import numpy as np
from libdvid import DVIDNodeService
from libdvid.voxels import DVID_BLOCK_WIDTH, VoxelsAccessor # See TODO, above

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transfer-cube-width-px', default=512, type=int)
    parser.add_argument('--roi', help="Name or URL of a roi instance")
    parser.add_argument('--subvol-bounds-zyx', help="Instead of providing --roi, use these bounds of a subregion to copy,"
                                                    " as a pair of tuples in ZYX order, in PIXEL COORDINATES,"
                                                    " such as [(0,0,0), (1024, 512, 512)" )
    parser.add_argument('source_instance_url', help='e.g. emdata1:7000/api/node/deadbeef/grayscale')
    parser.add_argument('destination_instance_url', help='e.g. emdata2:8000/api/node/abc123/grayscale')

    args = parser.parse_args()

    assert (args.roi is not None) ^ (args.subvol_bounds_zyx is not None), \
        "You must provide --roi OR --subvol_bounds-zyx (but not both)."

    subvol_bounds = None
    if args.subvol_bounds_zyx:
        subvol_bounds = eval(args.subvol_bounds_zyx)
        assert len(subvol_bounds) == 2, "Invalid value for --subvol_bounds_zyx"
        assert map(len, subvol_bounds) == [3,3], "Invalid value for --subvol_bounds_zyx"

    if args.transfer_cube_width_px % DVID_BLOCK_WIDTH != 0:
        sys.exit("The --transfer-cube-width-px must be a multiple of {}".format(DVID_BLOCK_WIDTH))

    copy_voxels( args.source_instance_url,
                 args.destination_instance_url,
                 args.transfer_cube_width_px,
                 args.roi,
                 subvol_bounds )


def copy_voxels( source_details,
                 destination_details,
                 transfer_cube_width_px=512,
                 roi=None,
                 subvol_bounds_zyx=None ):
    """
    Transfer voxels data from one DVID server to another.
    
    source_details:
        Either a tuple of (hostname, uuid, instance),
        or a url of the form http://hostname/api/node/uuid/instance
    
    destination_details:
        Same format as source_details, or just an instance name
        (in which case the destination is presumed to be in the same host/node as the source).
    
    transfer_cube_width_px:
        The data will be transferred one 'substack' at a time, with the given substack width.
    
    NOTE: Exactly ONE of the following parameters should be provided.
    
    roi:
        Same format as destination_details, but should point to a ROI instance.
    
    subvol_bounds_zyx:
        A tuple (start_zyx, stop_zyx) indicating a rectangular region to copy (instead of a ROI).
        Specified in pixel coordinates. Must be aligned to DVID block boundaries.
        For example: ((0,0,0), (1024, 1024, 512))
    """
    if isinstance(source_details, basestring):
        source_details = parse_instance_url( source_details )
    else:
        source_details = InstanceDetails(*source_details)
    src_accessor = VoxelsAccessor( *source_details )
    
    if isinstance(destination_details, basestring):
        destination_details = str_to_details( destination_details, default=source_details )
    else:
        destination_details = InstanceDetails(*destination_details)
    dest_accessor = VoxelsAccessor( *destination_details )

    assert (roi is not None) ^ (subvol_bounds_zyx is not None), \
        "You must provide roi OR subvol_bounds-zyx (but not both)."

    # Figure out what blocks ('substacks') we're copying
    if subvol_bounds_zyx:
        assert False, "User beware: The subvol_bounds_zyx option hasn't been tested yet. " \
                      "Now that you've been warned, comment out this assertion and give it a try. "\
                      "(It *should* work...)"

        assert len(subvol_bounds_zyx) == 2, "Invalid value for subvol_bounds_zyx"
        assert map(len, subvol_bounds_zyx) == [3,3], "Invalid value for subvol_bounds_zyx"

        subvol_bounds_zyx = np.array(subvol_bounds_zyx)
        subvol_shape = subvol_bounds_zyx[1] - subvol_bounds_zyx[0]
        np.array(subvol_bounds_zyx) / transfer_cube_width_px
        assert (subvol_shape % transfer_cube_width_px).all(), \
            "subvolume must be divisible by the transfer_cube_width_px"
        
        blocks_zyx = []
        transfer_block_indexes = np.ndindex( *(subvol_shape / transfer_cube_width_px) )
        for tbi in transfer_block_indexes:
            start_zyx = tbi*transfer_cube_width_px + subvol_bounds_zyx[0]
            blocks_zyx.append( SubstackZYX(transfer_cube_width_px, *start_zyx) )        
    elif roi is not None:
        if isinstance(roi, basestring):
            roi_details = str_to_details( roi, default=source_details )
        else:
            roi_details = InstanceDetails(*roi)
        roi_node = DVIDNodeService(roi_details.host, roi_details.uuid)
        blocks_zyx = roi_node.get_roi_partition(roi_details.instance, transfer_cube_width_px/DVID_BLOCK_WIDTH)[0]
    else:
        assert False

    # Fetch/write the blocks one at a time
    # TODO: We could speed this up if we used a threadpool...
    logger.debug( "Beginning Transfer of {} blocks ({} px each)".format( len(blocks_zyx), transfer_cube_width_px ) )
    for block_index, block_zyx in enumerate(blocks_zyx, start=1):
        start_zyxc = np.array(tuple(block_zyx[1:]) + (0,)) # skip item 0 ('size'), append channel
        stop_zyxc = start_zyxc + transfer_cube_width_px
        stop_zyxc[-1] = 1

        logger.debug("Fetching block: {} ({}/{})".format(start_zyxc[:-1], block_index, len(blocks_zyx)) )
        src_block_data = src_accessor.get_ndarray( start_zyxc, stop_zyxc )
        
        logger.debug("Writing block:  {} ({}/{})".format(start_zyxc[:-1], block_index, len(blocks_zyx)) )
        dest_accessor.post_ndarray( start_zyxc, stop_zyxc, new_data=src_block_data )
        
    logger.debug("DONE.")

InstanceDetails = collections.namedtuple('InstanceDetails', 'host uuid instance')
def str_to_details( url_or_name, default ):
    """
    Convert the given string into an InstanceDetails.
    If it's a url, parse the details directly.
    Otherwise, assume it's an instance name, and use the hostname/uuid provided in 'default'.
    """
    if '/api/node' in url_or_name:
        return InstanceDetails(*parse_instance_url(url_or_name))
    else:
        # Assume it's just a name, but with the default details
        return InstanceDetails(*(default[:-1] + (url_or_name,)))

def parse_instance_url(instance_url):
    """
    Parse a url of the form http://hostname/api/node/uuid/instance
    to an InstanceDetails tuple (host, uuid, instance)
    """
    url_format = "^(protocol://)?hostname/api/node/uuid/instance"
    for field in ['protocol', 'hostname', 'uuid', 'instance']:
        url_format = url_format.replace( field, '(?P<' + field + '>[^?]+)' )

    match = re.match( url_format, instance_url )
    if not match:
        raise RuntimeError('Did not understand node url: {}'.format( instance_url ))

    fields = match.groupdict()
    return InstanceDetails(fields['hostname'], fields['uuid'], fields['instance'])

if __name__ == "__main__":
    import sys
    logger.addHandler( logging.StreamHandler(sys.stdout) )
    logger.setLevel(logging.DEBUG)

    main()

import sys
import re
import logging
import argparse
import collections
from itertools import starmap

import numpy as np
import h5py

from libdvid import DVIDNodeService
from libdvid.voxels import VoxelsAccessor

SUBSTACK_SIZE = 512
DVID_BLOCK_SIZE = 32 # FIXME: This should be queried from the server, not hard-coded!

logger = logging.getLogger(__name__)

def download_to_h5( hostname, uuid, instance, roi, output_filepath, dset_name=None, compression='lzf'):
    """
    """
    ns = DVIDNodeService(hostname, uuid)
    va = VoxelsAccessor(hostname, uuid, instance, throttle=True)
    
    dset_name = dset_name or instance

    assert roi, "Must provide a ROI"
    logger.info("Downloading {hostname}/api/node/{uuid}/{instance}?roi={roi} to {output_filepath}/{dset_name}".format(**locals()))

    substacks, _packing_factor = ns.get_roi_partition(roi, SUBSTACK_SIZE / DVID_BLOCK_SIZE)

    # Substack tuples are (size, z, y, x)
    substacks_zyx = np.array(substacks)[:, 1:]
    roi_bb = ( np.min(substacks_zyx, axis=0),
               np.max(substacks_zyx, axis=0)+SUBSTACK_SIZE )
    
    with h5py.File(output_filepath, 'a') as output_file:
        try:
            del output_file[dset_name]
        except KeyError:
            pass
        
        dset = output_file.create_dataset( dset_name, shape=roi_bb[1], dtype=va.dtype, chunks=True, compression=compression )
    
        for i, substack_zyx in enumerate(substacks_zyx):
            logger.info("Substack {}/{} {}: Downloading...".format( i, len(substacks_zyx), list(substack_zyx) ))
            
            # Append a singleton channel axis
            substack_bb = np.array(( tuple(substack_zyx) + (0,),
                                     tuple(substack_zyx + SUBSTACK_SIZE) + (1,) ))
            
            # Includes singleton channel
            substack_data = va.get_ndarray(*substack_bb)

            logger.info("Substack {}/{} {}: Writing...".format( i, len(substacks_zyx), list(substack_zyx) ))
            dset[bb_to_slicing(*substack_bb[:,:-1])] = substack_data[...,0]

    logger.info("DONE Downloading {hostname}/api/node/{uuid}/{instance}?roi={roi} to {output_filepath}/{dset_name}".format(**locals()))

def bb_to_slicing(start, stop):
    """
    For the given bounding box (start, stop),
    return the corresponding slicing tuple.

    Example:
    
        >>> assert bb_to_slicing([1,2,3], [4,5,6]) == np.s_[1:4, 2:5, 3:6]
    """
    return tuple( starmap( slice, zip(start, stop) ) )
    
InstanceDetails = collections.namedtuple('InstanceDetails', 'host uuid instance')
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

def main():
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    
    # Read cmd-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-url", default="")
    parser.add_argument("--hostname", default="")
    parser.add_argument("--uuid", required=False, help="The node to download from.")
    parser.add_argument("--instance", required=False, help="The name of the data instance to modify. If it doesn't exist, it will be created first.")
    parser.add_argument("--roi", required=True)
    parser.add_argument("--compression", required=False)
    parser.add_argument("output_location", help="For example: /tmp/myfile.h5/dataset")
    args = parser.parse_args()

    if args.instance_url:
        assert not args.hostname, "Can't provide both hostname and url"
        assert not args.uuid, "Can't provide both uuid and url"
        assert not args.instance, "Can't provide both instance and url"
        hostname, uuid, instance =  parse_instance_url(args.instance_url)
    else:
        hostname, uuid, instance = args.hostname, args.uuid, args.instance

    filepath, dset_name = args.output_location.split('.h5')
    filepath += '.h5'
    dset_name = dset_name[1:] # drop leading '/'
    if not dset_name:
        sys.stderr.write("You must provide a dataset name, e.g. myfile.h5/mydataset\n")
        sys.exit(1)

    download_to_h5(hostname, uuid, instance, args.roi, filepath, dset_name, args.compression)

if __name__ == "__main__":
    # DEBUG
    #sys.argv += '--instance-url=http://emdata1:7000/api/node/3002/grayscale --roi=seven_column_roi --compression=lzf /tmp/seven-col.h5/grayscale'.split()
    
    main()

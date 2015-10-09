"""
Uploads a 3D hdf5 dataset to DVID using the ND-data REST API.

For usage:
    python upload-h5-to-dvid.py --help

Examples:
    python upload-h5-to-dvid.py /tmp/knott-tiny-sample.h5/grayscale
    python upload-h5-to-dvid.py --hostname=localhost:8000 --uuid=abcd1234 --data-name=grayscale /tmp/knott-tiny-sample.h5/grayscale
    python upload-h5-to-dvid.py --hostname=localhost:8000 --new-repo-alias=testrepo --data-name=grayscale /tmp/knott-tiny-sample.h5/grayscale    
"""
from __future__ import print_function
import sys
import os
import argparse

import numpy
import h5py

from libdvid import DVIDServerService, DVIDException
from libdvid.voxels import VoxelsAccessor, VoxelsMetadata

def main():
    # Read cmd-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", default="localhost:8000")
    parser.add_argument("--uuid", required=False, help="The node to upload to.  If not provided, a new repo will be created (see --new-repo-alias).")
    parser.add_argument("--data-name", required=False, help="The name of the data instance to modify. If it doesn't exist, it will be created first.")
    parser.add_argument("--new-repo-alias", required=False, help="If no uuid is provided, a new repo is created, with this name.")
    parser.add_argument("input_file", help="For example: /tmp/myfile.h5/dataset")
    args = parser.parse_args()

    if '.h5' not in args.input_file:
        sys.stderr.write("File name does not indicate hdf5.\n")
        sys.exit(1)

    filepath, dset_name = args.input_file.split('.h5')
    filepath += '.h5'
    if not dset_name:
        sys.stderr.write("You must provide a dataset name, e.g. myfile.h5/mydataset\n")
        sys.exit(1)

    if not os.path.exists(filepath):
        sys.stderr.write("File doesn't exist: {}\n".format(filepath))
        sys.exit(1)

    # If no uuid given, create a new repo on the server
    uuid = args.uuid
    if uuid is None:
        alias = args.new_repo_alias or "testrepo"
        server = DVIDServerService(args.hostname)
        uuid = server.create_new_repo(alias, "This is a test repo loaded with data from ".format(args.input_file))
        uuid = str(uuid)

    # Read the input data from the file
    print("Reading {}{}".format( filepath, dset_name ))
    with h5py.File(filepath) as f_in:
        data = f_in[dset_name][:]

    # We assume data is 3D or 4D, in C-order
    # We adjust it to 4D, fortran-order
    if data.ndim == 3:
        data = data[...,None]
    data = data.transpose()
    assert data.flags['F_CONTIGUOUS'], "Data is not contiguous!"
    assert data.ndim == 4, "Data must be 3D with axes zyx or 4D with axes zyxc (C-order)"
    assert data.shape[0] == 1, "Data must have exactly 1 channel, not {}".format( data.shape[0] )

    # Choose a default data instance name if necessary
    if data.dtype == numpy.uint8:
        data_name = args.data_name or "grayscale"
    elif data.dtype == numpy.uint64:
        data_name = args.data_name or "segmentation"
    else:
        sys.stderr.write("Unsupported dtype: {}\n".format(data.dtype))
        sys.exit(1)

    # Create the new data instance if it doesn't exist already        
    try:
        metadata = VoxelsAccessor.get_metadata(args.hostname, uuid, data_name)
        print("Data instance '{}' already exists.  Will update.".format( data_name ))
    except DVIDException:
        print("Creating new data instance: {}".format( data_name ))
        metadata = VoxelsMetadata.create_default_metadata(data.shape, data.dtype, 'cxyz', 1.0, 'nanometers')
        VoxelsAccessor.create_new(args.hostname, uuid, data_name, metadata)

    # Finally, push the data to the server
    print("Pushing data to {}".format( '{}/api/node/{}/{}'.format( args.hostname, uuid, data_name ) ))
    accessor = VoxelsAccessor(args.hostname, uuid, data_name)
    accessor.post_ndarray((0,0,0,0), data.shape, data)
    print("DONE.")

if __name__ == "__main__":
    main()

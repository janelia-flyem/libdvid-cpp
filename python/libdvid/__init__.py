from __future__ import absolute_import
from ._dvid_python import *

# There seems to be no easy way to attach
#  these member on the C++ side, and it's easy to just do it here.
@property
def DVIDException_status(self):
    return self.args[0]
DVIDException.status = DVIDException_status

@property
def DVIDException_message(self):
    return self.args[1]
DVIDException.message = DVIDException_message

from .mask_codec import encode_mask_blocks, decode_mask_blocks, encode_mask_array, decode_mask_array

def dissect_labelmap_block(instance_info, coord_xyz, supervoxels=False):
    """
    Debugging function for dissecting labelarray/labelmap blocks
    
    Returns:
        (num_labels, label_table, subblock_indices),
        
        In the DVID docs for labelarray compression, those are referred to as:
        N, 'packed labels', and 'label indices for subblocks', respectively.
        
        label_table is returned as a plain numpy array,
        whereas subblock_indices is returned as a dict with entries for each subblock:
            { (sbx,sby,sbz) : [index list] }
    """
    import getpass
    import numpy as np
    import requests, gzip

    server, uuid, instance = instance_info
    coord_xyz = np.array(coord_xyz)
    coord_str = '_'.join(map(str, coord_xyz))
    user = getpass.getuser()

    supervoxels = str(bool(supervoxels)).lower()
    url = f'http://{server}/api/node/{uuid}/{instance}/blocks/64_64_64/{coord_str}?supervoxels={supervoxels}&compression=blocks&u={user}&app=libdvid'
    r = requests.get(url)
    r.raise_for_status()

    # int32  Block 1 coordinate X (Note that this may not be starting block coordinate if it is unset.)
    # int32  Block 1 coordinate Y
    # int32  Block 1 coordinate Z
    # int32  # bytes for first block (N1)
    bx, by, bz, nbytes = np.frombuffer(r.content[:4*4], dtype=np.int32)
    assert ((bx, by, bz) == (coord_xyz // 64)).all()
    assert len(r.content) == 4*4 + nbytes
    block_data = gzip.decompress(r.content[4*4:])

    # 3 * uint32      values of gx, gy, and gz
    # uint32          # of labels (N), cannot exceed uint32.
    gx, gy, gz, num_labels = np.frombuffer(block_data[:4*4], dtype=np.int32)
    assert gx == gy == gz == 8

    # N * uint64      packed labels in little-endian format
    label_table_buffer = block_data[4*4 : (4*4 + 8*num_labels)]
    label_table = np.frombuffer(label_table_buffer, np.uint64)

    # Nsb * uint16  # of labels for sub-blocks.  Each uint16 Ns[i] = # labels for sub-block i.
    #               If Ns[i] == 0, the sub-block has no data (uninitialized), which
    #               is useful for constructing Blocks with sparse data.
    Nsb = gx * gy * gz
    subblock_start = 4*4 + 8*num_labels
    subblock_counts_buf = block_data[subblock_start:(subblock_start + Nsb*2)]
    subblock_counts = np.frombuffer(subblock_counts_buf, np.uint16)

    # Nsb * Ns * uint32   label indices for sub-blocks where Ns = sum of Ns[i] over all sub-blocks.
    #                     For each sub-block i, we have Ns[i] label indices of lBits.
    Ns = subblock_counts.sum()
    subblock_indices_buf = block_data[int(subblock_start + Nsb*2) : int(subblock_start + Nsb*2 + Nsb * Ns * 4)]
    subblock_indices_flat = np.frombuffer(subblock_indices_buf, np.uint32)

    subblock_indices = {}
    range_points = [0] + np.add.accumulate(subblock_counts).tolist()
    ranges = zip(range_points[:-1], range_points[1:])
    for (z, y, x), (start, stop) in zip(np.ndindex(gz, gy, gx), ranges):
        subblock_indices[(x,y,z)] = subblock_indices_flat[start:stop].tolist()

    return num_labels, label_table, subblock_indices

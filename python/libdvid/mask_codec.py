from itertools import starmap, product
import numpy as np
from skimage.util import view_as_blocks

SOLID_BACKGROUND = 0
SOLID_FOREGROUND = 1
MIXED = 2

def encode_mask_array(mask, corner=(0,0,0), foreground_label=1 ):
    """
    Compress a mask array using the DVID sparse encoding scheme.
    
    Note:
        If the array bounds are not block aligned, the array is padded first.
        In that case, the original shape of the array is lost.
        The caller is responsible for keeping track of it, if desired.
    
    Returns: bytes
    """
    assert mask.ndim == 3
    corner = np.asarray(corner)
    assert corner.shape == (3,)
    if (np.array(mask.shape) % 64).any():
        padding = 64 - (np.array(mask.shape) % 64)
        padded_mask = np.zeros( mask.shape + padding, np.bool )
        padded_mask[box_to_slicing((0,0,0), mask.shape)] = mask
    else:
        padded_mask = np.asarray(mask, np.bool, 'C')

    data_corners = np.array(list(ndrange((0,0,0), padded_mask.shape, (64,64,64))))
    blocks = ( padded_mask[box_to_slicing(corner, corner + 64)] for corner in data_corners )

    block_corners = ndrange(corner, corner + padded_mask.shape, (64,64,64))
    return encode_mask_blocks( blocks, block_corners, foreground_label )

def decode_mask_array( bytes_data, shape=None ):
    """
    Decode a mask array from the given bytes object.
    
    Args:
        shape: (Optional) Truncate the result to the given shape after decoding.
    
    Returns:
        mask, start_corner, foreground_label
    """
    blocks, corners, foreground_label = decode_mask_blocks(bytes_data)
    
    start_corner = np.min( corners, axis=0 )
    stop_corner = 64 + np.max(corners, axis=0)
    padded_shape = stop_corner - start_corner
    padded_mask = np.zeros( padded_shape, dtype=np.bool )

    block_corners = np.array(list(ndrange((0,0,0), padded_shape, (64,64,64))))
    for block, corner in zip(blocks, block_corners):
        padded_mask[box_to_slicing(corner, corner + 64)] = block

    if shape is not None and shape != padded_mask.shape:
        mask = padded_mask[box_to_slicing((0,0,0), shape)]
    else:
        mask = padded_mask
        
    return mask, start_corner, foreground_label

def encode_mask_blocks( blocks, block_corners=None, foreground_label=1 ):
    """
    Encode the given list of binary blocks, each with shape (64,64,64),
    into a bytes object using DVID's binary sparse blocks scheme.
    
    Args:
        blocks: Iterable of blocks (64,64,64)
        
        block_corners: (Optional.) List of coordinates.
        
        foreground_label: (Optional.) The DVID sparse blocks scheme allows you to
                          provide a uint64 label to associate with the binary mask.
    
    Returns: bytes
    """
    stream_contents = []
    stream_contents.append( np.array([8, 8, 8], np.uint32) )
    stream_contents.append( np.array([foreground_label], np.uint64) )

    if block_corners is None:
        block_corners = ndrange((0,0,0), (64,64,64*len(blocks)), (64,64,64))
    
    for block, corner in zip(blocks, block_corners):
        encoded_block_stream_contents = _encode_mask_block(block, corner)
        stream_contents.extend( encoded_block_stream_contents )

    stream_contents = np.concatenate(list(map(lambda a: a.view(np.uint8), stream_contents)))    
    encoded = stream_contents.tobytes()
    return encoded

def _encode_mask_block(block, block_corner=(0,0,0)):
    assert block.shape == (64,64,64)
    block = np.asarray(block, dtype=bool, order='C')

    stream_contents = []
    stream_contents.append( np.array(block_corner, np.int32) )

    block_sum = block.sum()
    if block_sum == 0:
        content_flag = SOLID_BACKGROUND
    if block_sum == 64*64*64:
        content_flag = SOLID_FOREGROUND
    else:
        content_flag = MIXED

    stream_contents.append(np.array([content_flag], dtype=np.uint8))

    if content_flag == MIXED:
        # In comments below,
        # 'Bx' == subblock ID; 'bx' == pixels within a subblock
        block_u64 =  block.view(np.uint64)
        subblocks_u64 = view_as_blocks( block_u64, (8,8,1) )

        assert block_u64.shape == (64,64,8)
        assert subblocks_u64.shape == (8,8,8,8,8,1) # (Bz, By, Bx, bz, by, 1)
        
        packed_subblocks = np.zeros(subblocks_u64.shape, np.uint8)
        
        for i in range(8):
            packed_subblocks[:] |= (subblocks_u64 >> (8*i - i))
        
        packed_subblocks = packed_subblocks.reshape(8*8*8, 8, 8, 1)
        for packed_subblock in packed_subblocks:
            block_sum = packed_subblock.sum()
            if block_sum == 0:
                stream_contents.append(np.array([SOLID_BACKGROUND], dtype=np.uint8))
                continue
            if block_sum == 8*8*255: # solid ones
                stream_contents.append(np.array([SOLID_FOREGROUND], dtype=np.uint8))
                continue
    
            stream_contents.append(np.array([MIXED], dtype=np.uint8))
            stream_contents.append( packed_subblock.flat[:] )

    return stream_contents


def decode_mask_blocks( bytes_data ):
    """
    Decode a bytes object from DVID's binary sparse volume format into a list of blocks and their starting coordinates.
    
    Args:
        bytes_data (bytes)    

    Returns:
        blocks: list of ndarray, shape=(64,64,64), dtype=bool
        corners: list of coordinates
        foreground_label: uint64
    """
    assert isinstance(bytes_data, bytes)
    n_bytes = len(bytes_data)
    next_byte_index = 0
    
    def extract_field(dtype):
        nonlocal next_byte_index
        width = dtype().nbytes
        field_buf = bytes_data[next_byte_index:next_byte_index+width]
        field = np.frombuffer( field_buf, dtype )[0]
        next_byte_index += width
        return field
    
    block_dims = (extract_field(np.uint32), extract_field(np.uint32), extract_field(np.uint32))
    assert block_dims == (8,8,8)
    
    foreground_label = extract_field(np.uint64)
    
    blocks = []
    corners = []
    
    while( next_byte_index < n_bytes ):
        block, corner, next_byte_index = _decode_mask_block( bytes_data, next_byte_index )
        blocks.append(block)
        corners.append(corner)
    
    return blocks, corners, foreground_label
        

def _decode_mask_block( bytes_data, next_byte_index ):

    def extract_field(dtype):
        nonlocal next_byte_index
        width = dtype().nbytes
        field_buf = bytes_data[next_byte_index:(next_byte_index + width)]
        field = np.frombuffer( field_buf, dtype )[0]
        next_byte_index += width
        return field

    corner = (extract_field(np.int32), extract_field(np.int32), extract_field(np.int32))
    content_flag = extract_field(np.uint8)
    
    if content_flag == SOLID_BACKGROUND:
        return (np.zeros((64,64,64), dtype=np.bool), corner, next_byte_index)
    
    if content_flag == SOLID_FOREGROUND:
        return (np.ones((64,64,64), dtype=np.bool), corner, next_byte_index)

    assert content_flag == MIXED
    block = np.zeros( (64,64,64), np.bool )
    subblocks = view_as_blocks(block, (8,8,8))
    subblocks_u64 = view_as_blocks( block.view(np.uint64), (8,8,1))
    assert subblocks_u64.shape == (8,8,8,8,8,1) # (Bz, By, Bx, bz, by, 1)

    # We're still using views, not copies
    assert is_view_of(subblocks, block)
    assert is_view_of(subblocks_u64, block)

    for Bz, By, Bx in np.ndindex(8,8,8):
        subblock = subblocks[Bz,By,Bx]
        subblock_u64 = subblocks_u64[Bz,By,Bx]

        content_flag = extract_field(np.uint8)
        if content_flag == SOLID_BACKGROUND:
            subblock[:] = 0
            continue
        if content_flag == SOLID_FOREGROUND:
            subblock[:] = 1
            continue
        
        assert content_flag == MIXED

        # Extract subblock bytestream
        bitstream_width = (8*8*8)//8
        subblock_bytes = bytes_data[next_byte_index:(next_byte_index + bitstream_width)]
        next_byte_index += bitstream_width

        packed_subblock = np.frombuffer(subblock_bytes, np.uint8).reshape(8,8,1).astype(np.uint64)
        assert packed_subblock.shape == subblock_u64.shape

        # Unpack
        for i in range(8):
            subblock_u64[:] |= (packed_subblock & (1<<i)) << (8*i - i)

    assert (view_as_blocks(block, (8,8,8)) == subblocks).all()
    return (block, corner, next_byte_index)
    
def is_view_of(view, base):
    if view is base:
            return True
    if view is None:
        return False
    return is_view_of(view.base, base)

def ndrange(start, stop=None, step=None):
    """
    Generator.

    Like np.ndindex, but accepts start/stop/step instead of
    assuming that start is always (0,0,0) and step is (1,1,1).
    
    Example:
    
    >>> for index in ndrange((1,2,3), (10,20,30), step=(5,10,15)):
    ...     print(index)
    (1, 2, 3)
    (1, 2, 18)
    (1, 12, 3)
    (1, 12, 18)
    (6, 2, 3)
    (6, 2, 18)
    (6, 12, 3)
    (6, 12, 18)
    """
    if stop is None:
        stop = start
        start = (0,)*len(stop)

    if step is None:
        step = (1,)*len(stop)

    assert len(start) == len(stop) == len(step), \
        f"tuple lengths don't match: ndrange({start}, {stop}, {step})"

    for index in product(*starmap(range, zip(start, stop, step))):
        yield index

def box_to_slicing(start, stop):
    """
    For the given bounding box (start, stop),
    return the corresponding slicing tuple.

    Example:
    
        >>> assert box_to_slicing([1,2,3], [4,5,6]) == np.s_[1:4, 2:5, 3:6]
    """
    return tuple( starmap( slice, zip(start, stop) ) )

if __name__ == "__main__":
    import lz4
    from DVIDSparkServices.util import Timer
    
    blocks = np.random.randint(0,2, size=(10,64,64,64), dtype=bool)
    #blocks = np.ones((10,64,64,64), dtype=bool)
    
    for block in blocks:
        # Randomly select a fourth of the subblocks to be completely 1,
        # and one fourth to be completely 0
        block_modes = np.random.randint(0,4, size=(8,8,8), dtype=int)
        v = view_as_blocks(block, (8,8,8))
        assert is_view_of(v, blocks)

        v &= (block_modes[..., None, None, None] != 0)
        v |= (block_modes[..., None, None, None] == 1)
    
    with Timer() as enc_timer:
        encoded = encode_mask_blocks(blocks)
    
    orig_bytes = 64*64*64*len(blocks)
    encoded_bytes = len(encoded)
    print(f"Size reduction: {orig_bytes} -> {encoded_bytes} ({orig_bytes/encoded_bytes:.1f}x)")

    with Timer() as dec_timer:
        decoded, corners, label = decode_mask_blocks(encoded)

    print(f"Mask encoding seconds: {enc_timer.seconds}")
    print(f"Mask decoding seconds: {dec_timer.seconds}")
    
    assert (np.array(decoded) == np.array(blocks)).all()
    
    with Timer() as lz4_enc_timer:
        lz4_encoded = lz4.compress(blocks)
        
    with Timer() as lz4_dec_timer:
        lz4_decoded = lz4.decompress(lz4_encoded)
    
    print(f"LZ4 Size reduction: {orig_bytes} -> {len(lz4_encoded)} ({orig_bytes/len(lz4_encoded):.1f}x)")
    
    print(f"LZ4 encoding seconds: {lz4_enc_timer.seconds}")
    print(f"LZ4 decoding seconds: {lz4_dec_timer.seconds}")
    
    print(f"Encoding slowdown (relative to LZ4): {(enc_timer.seconds / lz4_enc_timer.seconds):.3f}x")
    print(f"Decoding slowdown (relative to LZ4): {(dec_timer.seconds / lz4_dec_timer.seconds):.3f}x")

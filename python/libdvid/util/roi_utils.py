import numpy as np

from libdvid.voxels import DVID_BLOCK_WIDTH

def roi_blocks_for_box(start, stop):
    start, stop = np.array(start), np.array(stop)

    block_start = start // DVID_BLOCK_WIDTH
    block_stop = (stop+DVID_BLOCK_WIDTH-1) // DVID_BLOCK_WIDTH

    blocks_shape = block_stop - block_start
    blocks = np.array(list(np.ndindex(*blocks_shape)))
    blocks += block_start
    
    return blocks

def get_dilated_roi_blocks( node_service, roi_name, radius ):
    """
    Dilate a ROI by the given radius.
    (Radius is specified in blocks, not pixels.)
    """
    # Retrieve roi blocks
    block_coords = np.array( node_service.get_roi( roi_name ) )

    # Bounding box    
    min_block = np.min(block_coords, axis=0)
    max_block = np.max(block_coords, axis=0) + 1

    # Expand BB for the radius
    min_block_dilated = min_block - radius
    max_block_dilated = max_block + radius

    # Create an array to contain the roi mask (every pixel is one block)
    dilated_shape = max_block_dilated - min_block_dilated
    block_mask = np.zeros( dilated_shape, dtype=np.uint8 )

    # Write the mask
    offset_block_coords = block_coords - min_block_dilated
    block_mask[tuple( offset_block_coords.transpose() )] = 1

    # Dilate
    import vigra
    vigra.filters.multiBinaryDilation( block_mask, radius, out=block_mask )
    
    # Extract coordinates and un-offset
    dilated_block_coords = np.transpose( block_mask.nonzero() ) + min_block_dilated
    return dilated_block_coords

def get_eroded_roi_blocks( node_service, roi_name, radius ):
    """
    Erode a ROI by the given radius.
    (Radius is specified in blocks, not pixels.)
    """
    # Retrieve roi blocks
    block_coords = np.array( node_service.get_roi( roi_name ) )

    # Bounding box
    min_block = np.min(block_coords, axis=0)
    max_block = np.max(block_coords, axis=0) + 1

    # Expand BB by 1 because otherwise erosion has no zero
    # pixels on the border to use as the erosion value
    min_block_expanded = min_block - 1
    max_block_expanded = max_block + 1

    # Create an array to contain the roi mask (every pixel is one block)
    expanded_shape = max_block_expanded - min_block_expanded
    block_mask = np.zeros( expanded_shape, dtype=np.uint8 )
    
    # Write the mask
    offset_block_coords = block_coords - min_block_expanded
    block_mask[tuple( offset_block_coords.transpose() )] = 1

    # Erode
    import vigra
    vigra.filters.multiBinaryErosion( block_mask, radius, out=block_mask )
    
    # Extract coordinates and un-offset
    eroded_block_coords = np.transpose( block_mask.nonzero() ) + min_block_expanded
    return eroded_block_coords

if __name__ == '__main__':
    blocks = roi_blocks_for_box( (0, 33, 66 ), (1, 34, 129) )
    assert (blocks == [[0,1,2], [0,1,3], [0,1,4]]).all()

    # Medulla:
    # 
    # min_x=4700, max_x=5700
    # min_y=4300, max_y=5300
    # min_z=7000, max_z=11000

    # Kazunori's 1k x 1k x 4k test region
    medulla_test_roi_blocks = roi_blocks_for_box( (4704,      4320,      7008),
                                                  (4704+1024, 4320+1024, 7008+4*1024) )
    blocks = map(tuple, medulla_test_roi_blocks)
    print blocks

#     blocks = roi_blocks_for_box( (10*1024, 5*1024, 10*1024),
#                                  (12*1024, 7*1024, 12*1024) )
# 
#     # For testing purposes, remove the last GB from the roi
#     blocks = filter(lambda (x,y,z): not (x > 11*1024/32 and y > 6*1024/32 and z > 11*1024/32), blocks )
#     blocks = map(tuple, blocks)
#     print len(blocks)
# 
#     import libdvid
# 
#     print "Posting ROI..."    
#     node_service = libdvid.DVIDNodeService("localhost:8000", '9ace850166ee44f783aa9990f4edb926')
#     node_service.create_roi("perf_test_7gb")
#     node_service.post_roi("perf_test_7gb", blocks)
    
    print "DONE"

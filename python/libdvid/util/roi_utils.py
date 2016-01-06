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

if __name__ == '__main__':
    blocks = roi_blocks_for_box( (0, 33, 66 ), (1, 34, 129) )
    assert (blocks == [[0,1,2], [0,1,3], [0,1,4]]).all()

    blocks = roi_blocks_for_box( (10*1024, 5*1024, 10*1024),
                                 (12*1024, 7*1024, 12*1024) )

    # For testing purposes, remove the last GB from the roi
    blocks = filter(lambda (x,y,z): not (x > 11*1024/32 and y > 6*1024/32 and z > 11*1024/32), blocks )
    blocks = map(tuple, blocks)
    print len(blocks)

    import libdvid

    print "Posting ROI..."    
    node_service = libdvid.DVIDNodeService("localhost:8000", '9ace850166ee44f783aa9990f4edb926')
    node_service.create_roi("perf_test_7gb")
    node_service.post_roi("perf_test_7gb", blocks)
    
    print "DONE"

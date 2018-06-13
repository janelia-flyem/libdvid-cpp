import unittest

import numpy as np
from skimage.util import view_as_blocks

from libdvid import encode_mask_blocks, decode_mask_blocks, encode_mask_array, decode_mask_array
from libdvid.mask_codec import ndrange, box_to_slicing

class TestMaskCodec(unittest.TestCase):

    def _gen_test_volume(self):
        volume = np.random.randint(0,2, size=(2*64,64,3*64), dtype=bool)
        corner=(0,0,0)
        
        # Randomly select a fourth of the subblocks to be completely 1,
        # and one fourth to be completely 0
        block_corners = np.array(list(ndrange(corner, np.array(corner) + volume.shape, (64,64,64))))
        blocks = ( volume[box_to_slicing(corner, corner + 64)] for corner in block_corners )
        for block in blocks:
            block_modes = np.random.randint(0,4, size=(8,8,8), dtype=int)
            
            data_block = block.copy(order='C')
            v = view_as_blocks(data_block, (8,8,8))
    
            v &= (block_modes[..., None, None, None] != 0)
            v |= (block_modes[..., None, None, None] == 1)
        
            block[:] = data_block
        
        return volume

    def test_random_blocks(self):
        volume = self._gen_test_volume()
        blocks = view_as_blocks(volume, (64,64,64)).reshape(-1,64,64,64)

        encoded = encode_mask_blocks(blocks, [(1,2,3)]*len(blocks), 17)
        decoded, corners, label = decode_mask_blocks(encoded)
        
        assert label == 17
        assert (np.array(corners) == (1,2,3)).all()
        assert (np.array(decoded) == np.array(blocks)).all()

    def test_random_aligned_array(self):
        volume = self._gen_test_volume()

        encoded = encode_mask_array(volume, (1,2,3), 17)
        decoded, corner, label = decode_mask_array(encoded)
        
        assert label == 17
        assert (corner == (1,2,3)).all()
        assert (decoded == volume).all()

    def test_random_NON_aligned_array(self):
        aligned_volume = self._gen_test_volume()
        non_aligned_volume = aligned_volume[:-2, :, :-10]
        
        encoded = encode_mask_array(non_aligned_volume, (1,2,3), 17)
        decoded, corner, label = decode_mask_array(encoded, non_aligned_volume.shape)

        assert label == 17
        assert (corner == (1,2,3)).all()
        assert decoded.shape == non_aligned_volume.shape
        assert (decoded == non_aligned_volume).all()


if __name__ == "__main__":
    unittest.main()

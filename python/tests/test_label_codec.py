import unittest
import numpy as np
from libdvid import encode_label_block, decode_label_block

class TestLabelCodec(unittest.TestCase):
    
    def test_encode_solid_block(self):
        """
        A trivial test to verify that the header of
        the encoded data is correct for a solid block.
        """
        a = np.ones((64,64,64), np.uint64)
        a *= 13 # label 13 everywhere.
        encoded_data = encode_label_block(a)
        assert len(encoded_data) == 24
        assert (np.frombuffer(encoded_data[:12], np.uint32) == 8).all()
        assert np.frombuffer(encoded_data[12:16], np.uint32)[0] == 1
        assert np.frombuffer(encoded_data[16:24], np.uint64)[0] == 13


    def test_solid_block_round_trip(self):
        a = np.ones((64,64,64), np.uint64)
        a *= 13 # label 13 everywhere.
        encoded_data = encode_label_block(a)        
        decoded_block = decode_label_block(encoded_data)
        
        assert decoded_block.dtype == a.dtype
        assert decoded_block.shape == a.shape
        assert (decoded_block == a).all()


    def test_solid_subblock_round_trip(self):
        # Start with a random block
        a = np.random.randint(0,10, (64,64,64)).astype(np.uint64)
        
        # Change a subblock to a solid label.
        # (This is a special case in the spec.)
        a[:-8, -8, -8] = 14
        
        encoded_data = encode_label_block(a)
        decoded_block = decode_label_block(encoded_data)
        
        assert decoded_block.dtype == a.dtype
        assert decoded_block.shape == a.shape
        assert (decoded_block == a).all()


    def test_5_labels_round_trip(self):
        a = np.ones((64,64,64), np.uint64)

        n_labels = 5
        for i in range(n_labels):
            a.flat[i::n_labels] = i+13

        encoded_data = encode_label_block(a)
        decoded_block = decode_label_block(encoded_data)
        
        assert decoded_block.dtype == a.dtype
        assert decoded_block.shape == a.shape
        assert (decoded_block == a).all()


    def test_random_block_round_trip(self):
        a = np.random.randint(0,10, (64,64,64)).astype(np.uint64)

        encoded_data = encode_label_block(a)
        decoded_block = decode_label_block(encoded_data)
        
        assert decoded_block.dtype == a.dtype
        assert decoded_block.shape == a.shape
        assert (decoded_block == a).all()


if __name__ == "__main__":
    unittest.main()

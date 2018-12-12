import os
import time
import unittest
import contextlib
from datetime import timedelta

import numpy as np

from libdvid import encode_label_block, decode_label_block


BENCHMARKING = False

if BENCHMARKING:
    import os
    import lz4
    import gzip
    from libdvid.util import view_as_blocks
    import DVIDSparkServices
    ACTUAL_DATA_512_PATH = os.path.dirname(DVIDSparkServices.__file__) + '/../integration_tests/resources/labels.bin'
    
    @contextlib.contextmanager
    def Timer():
        result = _TimerResult()
        start = time.time()
        yield result
        result.seconds = time.time() - start
        result.timedelta = timedelta(seconds=result.seconds)
    
    class _TimerResult(object):
        seconds = -1.0
    
    def serialize_uint64_blocks(volume):
        """
        Compress and serialize a volume of uint64.
        
        Preconditions:
          - volume.dtype == np.uint64
          - volume.ndim == 3
          - volume.shape is divisible by 64
        
        Returns compressed_blocks, where the blocks are a flat list, in scan-order
        """
        assert volume.dtype == np.uint64
        assert volume.ndim == 3
        assert (np.array(volume.shape) % 64 == 0).all()
        
        block_view = view_as_blocks( volume, (64,64,64) )
        compressed_blocks = []
        for zi, yi, xi in np.ndindex(*block_view.shape[:3]):
            block = block_view[zi,yi,xi].copy('C')
            compressed_blocks.append( encode_label_block(block) )
            del block
        
        return compressed_blocks
    
    def deserialize_uint64_blocks(compressed_blocks, shape):
        """
        Reconstitute a volume that was serialized with serialize_uint64_blocks(), above.
        """
        volume = np.ndarray(shape, dtype=np.uint64)
        block_view = view_as_blocks( volume, (64,64,64) )
        
        for bi, (zi, yi, xi) in enumerate(np.ndindex(*block_view.shape[:3])):
            block = decode_label_block( compressed_blocks[bi] )
            block_view[zi,yi,xi] = block
        
        return volume
    
    HEADER_PRINTED = False
    def _test_block(labels, test_name):
        
        # labelarray
        with Timer() as timer:
            dvid_encoded_list = serialize_uint64_blocks(labels)
        dvid_encoded_bytes = sum(map(len, dvid_encoded_list))
        dvid_enc_time = timer.seconds
        dvid_enc_throughput = (labels.nbytes / dvid_enc_time) / 1e6
    
        with Timer() as timer:
            decoded = deserialize_uint64_blocks(dvid_encoded_list, labels.shape)
        assert (decoded == labels).all()
        dvid_dec_time = timer.seconds
        dvid_dec_throughput = (labels.nbytes / dvid_dec_time) / 1e6

        # DVID + gzip
        with Timer() as timer:
            gzipped_dvid_encoded_list = list(map(gzip.compress, dvid_encoded_list))
        gzipped_dvid_enc_time = timer.seconds + dvid_enc_time
        gzipped_dvid_enc_throughput = (labels.nbytes / gzipped_dvid_enc_time) / 1e6
        
        gzipped_dvid_encoded_bytes = sum(map(len, gzipped_dvid_encoded_list))
        print("+ GZIP:", gzipped_dvid_encoded_bytes)
        print(f"Compression ratio: {labels.nbytes/gzipped_dvid_encoded_bytes:.1f}x")
        print(f"DVID+GZIP encode throughput: {gzipped_dvid_enc_throughput} MB/s")
 
        with Timer() as timer:
            unzippped = list(map(gzip.decompress, gzipped_dvid_encoded_list))
        assert (decoded == labels).all()
        gzipped_dvid_dec_time = timer.seconds + dvid_dec_time
        gzipped_dvid_dec_throughput = (labels.nbytes / gzipped_dvid_dec_time) / 1e6
        print(f"DVID+GZIP decode throughput: {gzipped_dvid_dec_throughput} MB/s")

        # DVID + LZ4
        with Timer() as timer:
            lz4_dvid_encoded_list = list(map(lz4.compress, dvid_encoded_list))
        lz4_dvid_enc_time = timer.seconds + dvid_enc_time
        lz4_dvid_enc_throughput = (labels.nbytes / lz4_dvid_enc_time) / 1e6
        
        lz4_dvid_encoded_bytes = sum(map(len, lz4_dvid_encoded_list))
        print("+ LZ4:", lz4_dvid_encoded_bytes)
        print(f"Compression ratio: {labels.nbytes/lz4_dvid_encoded_bytes:.1f}x")
        print(f"DVID+LZ4 encode throughput: {lz4_dvid_enc_throughput} MB/s")
 
        with Timer() as timer:
            unzippped = list(map(lz4.decompress, lz4_dvid_encoded_list))
        assert (decoded == labels).all()
        lz4_dvid_dec_time = timer.seconds + dvid_dec_time
        lz4_dvid_dec_throughput = (labels.nbytes / lz4_dvid_dec_time) / 1e6
        print(f"DVID+LZ4 decode throughput: {lz4_dvid_dec_throughput} MB/s")

        # lz4
        with Timer() as timer:
            lz4_encoded = lz4.compress(labels)
        lz4_encoded_bytes = len(lz4_encoded)
        lz4_enc_time = timer.seconds
        lz4_enc_throughput = (labels.nbytes / lz4_enc_time) / 1e6
    
        with Timer() as timer:
            lz4_decoded = lz4.decompress(lz4_encoded)
        decoded_labels = np.frombuffer(lz4_decoded, np.uint64).reshape(labels.shape)
        assert (decoded_labels == labels).all()
        lz4_dec_time = timer.seconds
        lz4_dec_throughput = (labels.nbytes / lz4_dec_time) / 1e6

        
        global HEADER_PRINTED
        if not HEADER_PRINTED:
            print(f"{'':>20s} {'______ ENCODED BYTES ______ ':^41s}  | {'______ ENCODING TIME ______ ':^77s} | {'______ DECODING TIME ______ ':^77s} |")
            print(f"{'':>20s} {'LZ4':>10s} {'DVID':>10s} {'D+G':>10s} {'DECREASE':>9s} |"
                  f"{'------- LZ4 -------':>22s} {'------ DVID ------':>22s} {'---- DVID+GZIP ----':>22s} {'SLOWDOWN':>9s} |"
                  f"{'------- LZ4 -------':>22s} {'------ DVID ------':>22s} {'---- DVID+GZIP ----':>22s} {'SLOWDOWN':>9s} |")
            HEADER_PRINTED = True

        print(f"{test_name:>19s}: {lz4_encoded_bytes: 10d} {dvid_encoded_bytes: 10d} {lz4_encoded_bytes/dvid_encoded_bytes:8.1f}x |"
              f"{lz4_enc_time:6.2f}s ({lz4_enc_throughput:7.1f} MB/s) {dvid_enc_time:6.2f}s ({dvid_enc_throughput:7.1f} MB/s) {dvid_enc_time/lz4_enc_time:8.1f}x |"
              f"{lz4_dec_time:6.2f}s ({lz4_dec_throughput:7.1f} MB/s) {dvid_dec_time:6.2f}s ({dvid_dec_throughput:7.1f} MB/s) {dvid_dec_time/lz4_dec_time:8.1f}x |")
    
    def test_1px_stripes():
        shape = (512,512,512)
        #shape = (256,256,256)
        labels = np.indices(shape, np.uint16).sum(axis=0).astype(np.uint64)
        _test_block(labels, "1px stripes")
    
    def test_5px_stripes():
        shape = (512,512,512)
        #shape = (256,256,256)
        labels = np.indices(shape, np.uint16).sum(axis=0).astype(np.uint64) // 5
        _test_block(labels, "5px stripes")
    
    def test_10px_stripes():
        shape = (512,512,512)
        #shape = (256,256,256)
        labels = np.indices(shape, np.uint16).sum(axis=0).astype(np.uint64) // 10
        _test_block(labels, "10px stripes")
    
    def test_solid_subblocks():
        from libdvid.util import view_as_blocks
        shape = (512,512,512)
        #shape = (256,256,256)
        labels = np.zeros(shape, dtype=np.uint8)
        labels_view = view_as_blocks(labels, (64,64,64))
        labels_view[:] = np.random.randint(255, size=(8,8,8,1,1,1), dtype=np.uint8)
        assert (labels[0:64,0:64,0:64] == labels[0,0,0]).all()
        
        labels = labels.astype(np.uint64)
    
        _test_block(labels, "Solid sub-blocks")
    
    def test_actual_data():
        with open(ACTUAL_DATA_512_PATH, 'rb') as f:
            labels_bin = f.read()
        
        assert len(labels_bin) == 8 * 512**3
        
        labels = np.frombuffer(labels_bin, dtype=np.uint64).reshape(512,512,512)
        _test_block(labels, "Actual (old) data")

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
 
    def test_encode_zeros(self):
        """
        A trivial test to verify that the header of
        the encoded data is correct for a solid block of ZEROS
        """
        a = np.zeros((64,64,64), np.uint64)
        encoded_data = encode_label_block(a)
        assert len(encoded_data) == 24
        assert (np.frombuffer(encoded_data[:12], np.uint32) == 8).all()
        assert np.frombuffer(encoded_data[12:16], np.uint32)[0] == 1
        assert np.frombuffer(encoded_data[16:24], np.uint64)[0] == 0
     
    def test_completely_unique_voxels(self):
        """
        Tests the worst possible scenario:
        What if every voxel in the block has a unique label?
        """
        header_size = 16
        global_table_size = (64**3) * 8
        
        num_sublocks = 8**3
        voxels_per_subblock = 8**3 # 512 == 2**9
        
        bits_per_voxel = 9
        stream_bits_per_subblock = bits_per_voxel*voxels_per_subblock
        stream_bytes_per_subblock = (stream_bits_per_subblock + 7) // 8 # techinically, supposed to round up
                                                                        # (ok, in this case, it's divisible anyway)
        max_block_encoded_size = (   header_size
                                   + global_table_size
                                   + num_sublocks*2                             # num labels for each subblock
                                   + num_sublocks*voxels_per_subblock*4         # subblock tables
                                   + num_sublocks*stream_bytes_per_subblock )   # subblock bitstreams

        a = np.arange(64**3, dtype=np.uint32).reshape((64,64,64)) # every voxel is unique
        a = a.astype(np.uint64)

        encoded_data = encode_label_block(a)
        assert len(encoded_data) == max_block_encoded_size

        decoded = decode_label_block(encoded_data)
        assert (decoded == a).all()

    def test_tricky_block(self):
        test_inputs_dir = os.path.dirname(__file__) + '/../../tests/inputs'
        tricky_block = np.load(test_inputs_dir + '/tricky-label-block.npz')['block']
        encoded_data = encode_label_block(tricky_block)

        decoded = decode_label_block(encoded_data)
        assert (decoded == tricky_block).all()


if __name__ == "__main__":
    if BENCHMARKING:
        #test_solid_subblocks()
        #test_1px_stripes()
        #test_5px_stripes()
        #test_10px_stripes()
        test_actual_data()
    else:
        unittest.main()

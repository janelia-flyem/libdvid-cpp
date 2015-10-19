import unittest
import numpy
import json

from libdvid import DVIDNodeService, ConnectionMethod, Slice2D, BlockXYZ, SubstackXYZ, ErrMsg
from _test_utils import TEST_DVID_SERVER, get_testrepo_root_uuid, delete_all_data_instances

class Test_DVIDNodeService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.uuid = get_testrepo_root_uuid()
    
    @classmethod
    def tearDownClass(cls):
        delete_all_data_instances(cls.uuid)

    def test_custom_request(self):
        node_service = DVIDNodeService(TEST_DVID_SERVER, self.uuid)
        response_body = node_service.custom_request( "log", "", ConnectionMethod.GET )
         
        # This shouldn't raise an exception
        json_data = json.loads(response_body)
 
    def test_keyvalue(self):
        node_service = DVIDNodeService(TEST_DVID_SERVER, self.uuid)
        node_service.create_keyvalue("test_keyvalue")
        node_service.put("test_keyvalue", "kkkk", "vvvv")
        readback_value = node_service.get("test_keyvalue", "kkkk")
        self.assertEqual(readback_value, "vvvv")
  
        with self.assertRaises(ErrMsg):
            node_service.put("test_keyvalue", "kkkk", 123) # 123 is not a buffer.
 
    def test_grayscale_3d(self):
        node_service = DVIDNodeService(TEST_DVID_SERVER, self.uuid)
        node_service.create_grayscale8("test_grayscale_3d")
        data = numpy.random.randint(0, 255, (128,128,128)).astype(numpy.uint8).transpose()
        assert data.flags['F_CONTIGUOUS']
        node_service.put_gray3D( "test_grayscale_3d", data, (0,0,0) )
        retrieved_data = node_service.get_gray3D( "test_grayscale_3d", (30,31,32), (20,20,20) )
        self.assertTrue( (retrieved_data == data[20:50, 20:51, 20:52]).all() )
 
    def test_labels_3d(self):
        node_service = DVIDNodeService(TEST_DVID_SERVER, self.uuid)
        node_service.create_labelblk("test_labels_3d")
        data = numpy.random.randint(0, 2**63-1, (128,128,128)).astype(numpy.uint64).transpose()
        assert data.flags['F_CONTIGUOUS']
        node_service.put_labels3D( "test_labels_3d", data, (0,0,0) )
        retrieved_data = node_service.get_labels3D( "test_labels_3d", (30,31,32), (20,20,20) )
        self.assertTrue( (retrieved_data == data[20:50, 20:51, 20:52]).all() )
 
    def test_labels_3d_volsync(self):
        node_service = DVIDNodeService(TEST_DVID_SERVER, self.uuid)
        node_service.create_labelblk("test_labels_3d2", "test_labels_3d2_vol")
        data = numpy.random.randint(0, 2**63-1, (128,128,128)).astype(numpy.uint64).transpose()
        assert data.flags['F_CONTIGUOUS']
        node_service.put_labels3D( "test_labels_3d2", data, (0,0,0) )
        retrieved_data = node_service.get_labels3D( "test_labels_3d2", (30,31,32), (20,20,20) )
        self.assertTrue( (retrieved_data == data[20:50, 20:51, 20:52]).all() )
 
    @unittest.skip("FIXME: No way to create tile data via the DVID http API.")
    def test_grayscale_2d_tile(self):
        # Create tile data here...
 
        # Now retrieve a tile.
        retrieved_tile = node_service.get_tile_slice( "test_grayscale_2d_tile", Slice2D.XY, 0, (0,0,0) )
     
    def test_roi(self):
        node_service = DVIDNodeService(TEST_DVID_SERVER, self.uuid)
        node_service.create_roi("test_roi")
        node_service.post_roi("test_roi", [(1,2,3),(2,3,4),(4,5,6)])
        roi_blocks = node_service.get_roi("test_roi")
        self.assertEqual(roi_blocks, [(1,2,3),(2,3,4),(4,5,6)])

    def test_roi_3d(self):
        node_service = DVIDNodeService(TEST_DVID_SERVER, self.uuid)
        node_service.create_roi("test_roi_3d")
        
        # Create an upside-down L-shaped roi in the first 4 blocks:
        # 
        # 1 1
        # 1 0
        node_service.post_roi("test_roi_3d", [(0,0,0),(1,0,0),(0,1,0)])

        expected_data = numpy.ones((64,64,32), dtype=numpy.uint8, order='F')
        expected_data[32:, 32:] = 0

        retrieved_data = node_service.get_roi3D( "test_roi_3d", (64,64,32), (0,0,0) )
        self.assertEqual( retrieved_data.shape, expected_data.shape )
        self.assertTrue( (retrieved_data == expected_data).all() )

    def test_roi_partition(self):
        node_service = DVIDNodeService(TEST_DVID_SERVER, self.uuid)
        node_service.create_roi("test_roi_partition")
         
        blocks = [(0,0,0),(1,1,1),(2,2,2),(3,3,3)]
        node_service.post_roi("test_roi_partition", blocks)
        substacks, packing_factor = node_service.get_roi_partition("test_roi_partition", 4)
         
        self.assertEqual(substacks, [SubstackXYZ(0,0,0,4*32)])        
        self.assertEqual( packing_factor, float(len(blocks))/(len(substacks) * 4**3) )
 
        blocks += [(4,0,0)]
        node_service.post_roi("test_roi_partition", blocks)
        substacks, packing_factor = node_service.get_roi_partition("test_roi_partition", 4)
  
        self.assertEqual(substacks, [SubstackXYZ(0,0,0,4*32), SubstackXYZ(128,0,0,4*32)])
  
        # FIXME: DVID returns "NumActiveBlocks: 8" here, even though there should only be 5.
        #        That's wack, right?
        #  self.assertEqual( packing_factor, float(len(blocks))/(len(substacks) * 4**3) )
 
    def test_roi_ptquery(self):
        node_service = DVIDNodeService(TEST_DVID_SERVER, self.uuid)
        node_service.create_roi("test_roi")
        node_service.post_roi("test_roi", [(1,2,3),(2,3,4),(4,5,6)])
        query_results = node_service.roi_ptquery( "test_roi", [(0,0,0), (32, 64, 32*3)] )
        self.assertEqual( query_results, [False, True] )        

if __name__ == "__main__":
    unittest.main()

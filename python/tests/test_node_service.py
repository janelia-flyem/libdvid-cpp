import unittest
import numpy
import json

from libdvid import DVIDNodeService, ConnectionMethod

TEST_SERVER = "127.0.0.1:8000"

class Test_DVIDNodeService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # FIXME: Create the repo and uuid 
        cls.uuid = "87f0a170155f4b54933cc879346f04e6"

    @classmethod
    def tearDownClass(cls):
        # TODO: clean up from setUpClass. 
        pass

    def test_custom_request(self):
        node_service = DVIDNodeService(TEST_SERVER, self.uuid)
        response_body = node_service.custom_request( "log", "", ConnectionMethod.GET)
        
        # This shouldn't raise an exception
        json_data = json.loads(response_body)

    def test_keyvalue(self):
        node_service = DVIDNodeService(TEST_SERVER, self.uuid)
        node_service.create_keyvalue("stuart_keyvalue")
        node_service.put("stuart_keyvalue", "kkkk", "vvvv")
        readback_value = node_service.get("stuart_keyvalue", "kkkk")
        self.assertEqual(readback_value, "vvvv")
 
        with self.assertRaises(RuntimeError):
            node_service.put("stuart_keyvalue", "kkkk", 123) # 123 is not a buffer.

    def test_grayscale8(self):
        node_service = DVIDNodeService(TEST_SERVER, self.uuid)
        node_service.create_grayscale8("stuart_grayscale")
        data = numpy.random.randint(0, 255, (128,128,128)).astype(numpy.uint8)
        node_service.put_gray3D( "stuart_grayscale", data, (0,0,0) )
        retrieved_data = node_service.get_gray3D( "stuart_grayscale", (30,30,30), (20,20,20) )
        self.assertTrue( (retrieved_data == data[20:50, 20:50, 20:50]).all() )

    def test_labels64(self):
        node_service = DVIDNodeService(TEST_SERVER, self.uuid)
        node_service.create_labelblk("stuart_labelblk")
        data = numpy.random.randint(0, 2**63-1, (128,128,128)).astype(numpy.uint64)
        node_service.put_labels3D( "stuart_labelblk", data, (0,0,0) )
        retrieved_data = node_service.get_labels3D( "stuart_labelblk", (30,30,30), (20,20,20) )
        self.assertTrue( (retrieved_data == data[20:50, 20:50, 20:50]).all() )

if __name__ == "__main__":
    unittest.main()

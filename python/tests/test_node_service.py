import unittest
from libdvid import DVIDNodeService 

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

    def test_keyvalue(self):
        node_service = DVIDNodeService(TEST_SERVER, self.uuid)
        node_service.create_keyvalue("stuart_keyvalue")
        node_service.put("stuart_keyvalue", "kkkk", "vvvv")
        readback_value = node_service.get("stuart_keyvalue", "kkkk")
        self.assertEqual(readback_value, "vvvv")

        with self.assertRaises(RuntimeError):
            node_service.put("stuart_keyvalue", "kkkk", 123) # 123 is not a buffer.
                
if __name__ == "__main__":
    unittest.main()

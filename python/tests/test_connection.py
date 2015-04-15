import unittest
import json
from libdvid import DVIDConnection, ConnectionMethod

TEST_SERVER = "127.0.0.1:8000"

class Test_DVIDConnection(unittest.TestCase):

    def test_get(self):
        connection = DVIDConnection(TEST_SERVER)
        status, body, error_message = connection.make_request( "/server/info", ConnectionMethod.GET);
        self.assertEqual(status, 200)
        self.assertEqual(error_message, "")
        
        # This shouldn't raise an exception
        json_data = json.loads(body)

    def test_garbage_request(self):
        connection = DVIDConnection(TEST_SERVER)
        status, body, error_message = connection.make_request( "/does/not/exist", ConnectionMethod.GET);
        
        # No connection errors, just missing data.
        self.assertEqual(error_message, "")

        self.assertEqual(status, 404)
        self.assertNotEqual(body, "")
        

if __name__ == "__main__":
    unittest.main()

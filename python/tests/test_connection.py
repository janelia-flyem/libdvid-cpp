import unittest
import json
from libdvid import DVIDConnection, ConnectionMethod, DVIDException
from _test_utils import TEST_DVID_SERVER

class Test_DVIDConnection(unittest.TestCase):

    def test_default_user(self):
        # For backwards compatibility, make sure we can create a
        # DVIDNodeService without supplying a user name
        connection = DVIDConnection(TEST_DVID_SERVER)

    def test_get(self):
        connection = DVIDConnection(TEST_DVID_SERVER, "me@foo.com", "myapp")
        status, body, error_message = connection.make_request( "/server/info", ConnectionMethod.GET);
        self.assertEqual(status, 200)
        self.assertEqual(error_message, "")
        
        # This shouldn't raise an exception
        json_data = json.loads(body)

    def test_garbage_request(self):
        connection = DVIDConnection(TEST_DVID_SERVER, "me2@bar.com", "myapp2")
        status, body, error_message = connection.make_request( "/does/not/exist", ConnectionMethod.GET, checkHttpErrors=False);
        
        # No connection errors, just missing data.
        self.assertEqual(error_message, "")

        self.assertEqual(status, 404)
        self.assertNotEqual(body, "")

    def test_garbage_request_throws(self):
        connection = DVIDConnection(TEST_DVID_SERVER, "me2@bar.com", "myapp2")
        
        try:
            status, body, error_message = connection.make_request( "/does/not/exist", ConnectionMethod.GET);
        except DVIDException as ex:
            assert ex.status == 404
        else:
            assert False, "Bad requests are supposed to throw exceptions!"

if __name__ == "__main__":
    unittest.main()

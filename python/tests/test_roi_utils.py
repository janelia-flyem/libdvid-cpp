import unittest

from libdvid.util.roi_utils import copy_roi

from libdvid import DVIDNodeService, ConnectionMethod, Slice2D, BlockZYX, SubstackZYX, PointZYX, ErrMsg
from _test_utils import TEST_DVID_SERVER, get_testrepo_root_uuid, delete_all_data_instances


class Test_roi_utils(unittest.TestCase):
    

    @classmethod
    def setUpClass(cls):
        cls.uuid = get_testrepo_root_uuid()

        node_service = DVIDNodeService(TEST_DVID_SERVER, cls.uuid, "unittest@test_roi_utils.py", "Test_roi_utils")
        node_service.create_roi("src_roi")
        node_service.post_roi("src_roi", [(1,2,3),(2,3,4),(4,5,6)])
        roi_blocks = node_service.get_roi("src_roi")
    
    def test_copy_roi(self):
        expected_blocks = [(1,2,3),(2,3,4),(4,5,6)]

        # We aren't really exercising transferring between servers here.  Too bad.
        src_info  = (TEST_DVID_SERVER, self.uuid, 'src_roi')
        dest_info = (TEST_DVID_SERVER, self.uuid, 'dest_roi')

        node_service = DVIDNodeService(TEST_DVID_SERVER, self.uuid, "unittest@test_roi_utils.py", "Test_roi_utils")
        written_blocks = node_service.get_roi("src_roi")
        assert set(expected_blocks) == set(written_blocks)

if __name__ == "__main__":
    unittest.main()

import unittest
from itertools import starmap

import numpy

from libdvid import DVIDNodeService
from libdvid.voxels import VoxelsAccessor, VoxelsMetadata

from _test_utils import TEST_DVID_SERVER, get_testrepo_root_uuid, delete_all_data_instances

def roi_to_slice(start, stop):
    return tuple( starmap(slice, zip(start, stop)) )

UUID = get_testrepo_root_uuid()

class TestVoxelsAccessor(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Override.  Called by nosetests.
        """
        # Choose names
        cls.dvid_repo = "datasetA"
        cls.data_name = "indices_data"
        cls.volume_location = "/repos/{dvid_repo}/volumes/{data_name}".format( **cls.__dict__ )

        cls.data_uuid = get_testrepo_root_uuid()
        cls.node_location = "/repos/{dvid_repo}/nodes/{data_uuid}".format( **cls.__dict__ )

        # Generate some test data
        data = numpy.random.randint(0, 255, (1, 128, 256, 512))
        data = numpy.asfortranarray(data, numpy.uint8)
        cls.original_data = data
        cls.voxels_metadata = VoxelsMetadata.create_default_metadata(data.shape, data.dtype, "cxyz", 1.0, "")

        # Write it to a new data instance
        node_service = DVIDNodeService(TEST_DVID_SERVER, cls.data_uuid)
        node_service.create_grayscale8(cls.data_name)
        node_service.put_gray3D( cls.data_name, data[0,...], (0,0,0) )

    @classmethod
    def tearDownClass(cls):
        """
        Override.  Called by nosetests.
        """
        delete_all_data_instances(cls.data_uuid)    
 
    def test_get_ndarray(self):
        """
        Get some data from the server and check it.
        """
        start, stop = (0,9,5,50), (1,10,20,150)
        dvid_vol = VoxelsAccessor( TEST_DVID_SERVER, self.data_uuid, self.data_name )
        subvolume = dvid_vol.get_ndarray( start, stop )
        assert (self.original_data[roi_to_slice(start, stop)] == subvolume).all()

    def test_get_ndarray_throttled(self):
        """
        Get some data from the server and check it.
        Enable throttle with throttle=True
         
        Note: This test doesn't really exercise our handling of 503 responses...
        """
        start, stop = (0,9,5,50), (1,10,20,150)
        dvid_vol = VoxelsAccessor( TEST_DVID_SERVER, self.data_uuid, self.data_name, throttle=True )
        subvolume = dvid_vol.get_ndarray( start, stop )
        assert (self.original_data[roi_to_slice(start, stop)] == subvolume).all()

    def test_get_ndarray_throttled_2(self):
        """
        Get some data from the server and check it.
        Enable throttle via query_args
 
        Note: This test doesn't really exercise our handling of 503 responses...
        """
        start, stop = (0,9,5,50), (1,10,20,150)
        dvid_vol = VoxelsAccessor( TEST_DVID_SERVER, self.data_uuid, self.data_name, query_args={'throttle' : 'on'} )
        subvolume = dvid_vol.get_ndarray( start, stop )
        assert (self.original_data[roi_to_slice(start, stop)] == subvolume).all()
 
#     def test_get_ndarray_throttled_3(self):
#         """
#         Get some data from the server and check it.
# 
#         Note: This test doesn't really exercise our handling of 503 responses.
#               When this class was based on pydvid, we used the 'busy_count' feature of h5mockserver 
#               to force 503 responses. Does DVID have a similar testing feature?
#         """
#         start, stop = (0,9,5,50), (1,10,20,150)
#         dvid_vol = VoxelsAccessor( TEST_DVID_SERVER, self.data_uuid, self.data_name, throttle=True, retry_timeout=7.0, warning_interval=3.0 )
#         subvolume = dvid_vol.get_ndarray( start, stop )
#         assert (self.original_data[roi_to_slice(start, stop)] == subvolume).all()
      
    def test_post_ndarray(self):
        """
        Modify a remote subvolume and verify that the server wrote it.
        """
        # Cutout dims
        start, stop = (0,0,32,64), (1,32,64,96)
        shape = numpy.subtract( stop, start )
   
        # Generate test data
        new_subvolume = numpy.random.randint( 0,1000, shape ).astype( numpy.uint8 )
        new_subvolume = numpy.asfortranarray(new_subvolume, numpy.uint8)
   
        # Send to server
        dvid_vol = VoxelsAccessor( TEST_DVID_SERVER, self.data_uuid, self.data_name )
        dvid_vol.post_ndarray(start, stop, new_subvolume)

        # Now read it back
        read_subvolume = dvid_vol.get_ndarray( start, stop )
        assert (read_subvolume == new_subvolume).all()

        # Modify our master copy so other tests don't get messed up.
        self.original_data[roi_to_slice(start, stop)] = new_subvolume

   
    def test_post_slicing(self):
        # Cutout dims
        start, stop = (0,0,32,64), (1,32,64,96)
        shape = numpy.subtract( stop, start )
   
        # Generate test data
        new_subvolume = numpy.random.randint( 0,1000, shape ).astype( numpy.uint8 )
        new_subvolume = numpy.asfortranarray(new_subvolume, numpy.uint8)
   
        # Send to server
        dvid_vol = VoxelsAccessor( TEST_DVID_SERVER, self.data_uuid, self.data_name )
        dvid_vol[roi_to_slice(start, stop)] = new_subvolume

        # Now read it back
        read_subvolume = dvid_vol.get_ndarray( start, stop )
        assert (read_subvolume == new_subvolume).all()

        # Modify our master copy so other tests don't get messed up.
        self.original_data[roi_to_slice(start, stop)] = new_subvolume
  
    def test_post_reduced_dim_slicing(self):
        # Cutout dims
        start, stop = (0,0,32,64), (1,32,64,96)
        shape = numpy.subtract( stop, start )
   
        # Generate test data
        new_subvolume = numpy.random.randint( 0,1000, shape ).astype( numpy.uint8 )
        new_subvolume = numpy.asfortranarray(new_subvolume, numpy.uint8)
   
        # Send to server
        dvid_vol = VoxelsAccessor( TEST_DVID_SERVER, self.data_uuid, self.data_name )
        dvid_vol[0, 0:32, 32:64, 64:96] = new_subvolume[0,...]

        # Now read it back
        read_subvolume = dvid_vol.get_ndarray( start, stop )
        assert (read_subvolume == new_subvolume).all()

        # Modify our master copy so other tests don't get messed up.
        self.original_data[roi_to_slice(start, stop)] = new_subvolume
  
    def test_get_full_volume_via_slicing(self):
        dvid_vol = VoxelsAccessor( TEST_DVID_SERVER, self.data_uuid, self.data_name )

        # Two slicing syntaxes for the same thing
        subvolume1 = dvid_vol[:]
        subvolume2 = dvid_vol[...]

        # Results should match
        assert (subvolume1 == self.original_data).all()
        assert (subvolume1 == subvolume2).all()         
   
    def test_get_full_slicing(self):
        dvid_vol = VoxelsAccessor( TEST_DVID_SERVER, self.data_uuid, self.data_name )
        full_slicing = roi_to_slice( (0,)*4, self.original_data.shape)
        subvolume = dvid_vol[full_slicing]
        assert (subvolume == self.original_data).all()
       
    def test_get_partial_slicing(self):
        dvid_vol = VoxelsAccessor( TEST_DVID_SERVER, self.data_uuid, self.data_name )
        full_slicing = roi_to_slice( (0,)*4, self.original_data.shape)
        partial_slicing = full_slicing[:-1]
        subvolume = dvid_vol[partial_slicing]
        assert (subvolume == self.original_data).all()
  
    def test_get_ellipsis_slicing(self):
        dvid_vol = VoxelsAccessor( TEST_DVID_SERVER, self.data_uuid, self.data_name )
        full_slicing = roi_to_slice( (0,)*4, self.original_data.shape)
        partial_slicing = full_slicing[:-2] + (Ellipsis,) + full_slicing[-1:]
        subvolume = dvid_vol[partial_slicing]
        assert (subvolume == self.original_data).all()
  
    def test_get_reduced_dim_slicing(self):
        # Retrieve from server
        dvid_vol = VoxelsAccessor( TEST_DVID_SERVER, self.data_uuid, self.data_name )
        assert self.original_data.shape == (1, 128, 256, 512), "Update this unit test."
        full_roi = ((0,0,0,0), (1, 128, 256, 512))        
        subvol_roi = ((0,0,10,0), (1, 128, 11, 512))
        
        reduced_subvol_roi = ((0,0,0), (1, 128, 512))
        reduced_dim_slicing = numpy.s_[0:1, 0:128, 10, 0:512] # Notice that the third dim is dropped
        
        # request
        subvolume = dvid_vol[reduced_dim_slicing]
          
        # Check dimensionality/shape of returned volume
        reduced_shape = numpy.subtract(reduced_subvol_roi[1], reduced_subvol_roi[0])
        assert subvolume.shape == tuple(reduced_shape)

        # Before we compare, re-insert the dropped axis
        assert (subvolume == self.original_data[reduced_dim_slicing]).all()
  
    def test_get_channel_slicing(self):
        """
        Test that slicing in the channel dimension works.
        This is a special case because the entire volume needs to be requested from DVID, 
        but only the requested subset of channels will be returned.
        
        FIXME: libdvid only supports single-channel dtypes right now anyway, so this test doesn't do anything...
        """
        # Retrieve from server
        dvid_vol = VoxelsAccessor( TEST_DVID_SERVER, self.data_uuid, self.data_name )
        subvolume = dvid_vol[0:1, 9:10, 5:20, 50:150]
          
        # Compare
        assert (subvolume == self.original_data[0:1, 9:10, 5:20, 50:150]).all()
  
    def test_get_stepped_slicing(self):
        """
        """
        # Retrieve from server
        dvid_vol = VoxelsAccessor( TEST_DVID_SERVER, self.data_uuid, self.data_name )
        subvolume = dvid_vol[0:1, 1:10:3, 5:20:5, 50:150:10]
          
        # Compare to file
        full_start = (0,) * len( self.original_data.shape )
        full_stop = self.original_data.shape
        stored_stepped_volume = self.original_data[0:1, 1:10:3, 5:20:5, 50:150:10]
  
        assert subvolume.shape == stored_stepped_volume.shape
        assert subvolume.dtype == stored_stepped_volume.dtype
        assert (subvolume == stored_stepped_volume).all()
  
    def test_extra_query_args(self):
        """
        Create a VoxelsAccessor that uses extra query args 
        They come after the '?' in the REST URI.  For example:
        http://localhost/api/node/mydata/_0_1_2/10_10_10/0_0_0?roi=whatever&attenuation=3
        """
        # Retrieve from server
        start, stop = (0,9,5,50), (1,10,20,150)
        query_args = {'roi' : 'some_ref', 'attenuation' : 5}
        dvid_vol = VoxelsAccessor( TEST_DVID_SERVER, self.data_uuid, self.data_name, query_args=query_args )
        subvolume = dvid_vol.get_ndarray( start, stop )
        
        # Compare
        assert (subvolume == self.original_data[roi_to_slice(start, stop)]).all()
            
    def test_zy_post_negative_coordinates(self):
        """
        Just make sure nothing blows up if we post to negative coordinates.
        """
        # Cutout dims (must be block-aligned for the POST)
        start, stop = (0,-32,0,-64), (1,32,32,128)
        shape = numpy.subtract( stop, start )
   
        # Generate test data
        subvolume = numpy.random.randint( 0,1000, shape ).astype(numpy.uint8)
        subvolume = numpy.asfortranarray(subvolume, numpy.uint8)
 
        dvid_vol = VoxelsAccessor( TEST_DVID_SERVER, self.data_uuid, self.data_name )
 
        # Send to server
        dvid_vol.post_ndarray(start, stop, subvolume)
         
        # Now try to 'get' data from negative coords
        read_back_vol = dvid_vol.get_ndarray(start, stop)
        assert (read_back_vol == subvolume).all()
 
    def test_zz_quickstart_usage(self):
        import json
        import numpy
        from libdvid import DVIDConnection, ConnectionMethod
        from libdvid.voxels import VoxelsAccessor, VoxelsMetadata
           
        # Open a connection to DVID
        connection = DVIDConnection( "localhost:8000" )
          
        # Get detailed dataset info: /api/repos/info (note: /api is prepended automatically)
        status, body, error_message = connection.make_request( "/repos/info", ConnectionMethod.GET)
        dataset_details = json.loads(body)
        # print json.dumps( dataset_details, indent=4 )
          
        # Create a new remote volume (assuming you already know the uuid of the node)
        uuid = UUID
        voxels_metadata = VoxelsMetadata.create_default_metadata( (1,0,0,0), numpy.uint8, 'cxyz', 1.0, "" )
        VoxelsAccessor.create_new( "localhost:8000", uuid, "my_volume", voxels_metadata )
  
        # Use the VoxelsAccessor convenience class to manipulate a particular data volume     
        accessor = VoxelsAccessor( "localhost:8000", uuid, "my_volume" )
        # print dvid_volume.axiskeys, dvid_volume.dtype, dvid_volume.minindex, dvid_volume.shape
           
        # Add some data (must be block-aligned)
        # Must include all channels.
        # Must be FORTRAN array, using FORTRAN indexing order conventions
        # (Use order='F', and make sure you're indexing it as cxyz)
        updated_data = numpy.ones( (1,128,192,256), dtype=numpy.uint8, order='F' )
        updated_data = numpy.asfortranarray(updated_data)
        accessor[:, 0:128, 32:224, 256:512] = updated_data
        # OR:
        #accessor.post_ndarray( (0,10,20,30), (1,110,120,130), updated_data )
          
        # Read from it (First axis is channel.)
        cutout_array = accessor[:, 10:110, 40:120, 300:330]
        # OR:
        cutout_array = accessor.get_ndarray( (0,10,40,300), (1,110,120,330) )
  
        assert isinstance(cutout_array, numpy.ndarray)
        assert cutout_array.shape == (1,100,80,30)

if __name__ == "__main__":
    unittest.main()


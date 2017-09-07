import json

import numpy
#import jsonschema

try:
    import vigra
    _have_vigra = True
except ImportError:
    _have_vigra = False
    
try:
    import h5py
    _have_h5py = True
except:
    _have_h5py = False

# import pydvid.util
# metadata_schema = pydvid.util.parse_schema( 'dvid-voxels-metadata-v0.02.schema.json' )

class VoxelsMetadata(dict):
    """
    A dict subclass for the dvid nd-data metadata response.
    Also provides the following convenience attributes: ``minindex``, ``shape``, ``dtype``, ``axiskeys``
    """
    
    @property
    def shape(self):
        """
        Property.  The maximum coordinates in the DVID volume coordinate space.
        This is the stop coordinate of the volume's bounding box.
        All data above this coordinate in any dimension is guaranteed to be invalid.
        """
        return self._shape

    @shape.setter
    def shape(self, new_shape):
        assert new_shape[-1] == self.shape[-1], "Can't change the number of channels."
        
        # Update JSON data to match
        for axisinfo, new_axis_max, axis_min in zip(self["Axes"], new_shape[:-1][::-1], self.minindex[:-1][::-1]):
            axisinfo["Size"] = new_axis_max - axis_min

        self._shape = new_shape
    @property
    def minindex(self):
        """
        Property.  The starting coordinate of the volume's bounding box.
        All data below this coordinate in any dimension is guaranteed to be invalid.
        """
        return self._minindex

    @minindex.setter
    def minindex(self, new_minindex):
        assert new_minindex[-1] == self.minindex[-1], "Can't change the number of channels."

        # Update JSON data to match
        for axisinfo, new_axis_min, axis_shape in zip(self["Axes"], new_minindex[:-1][::-1], self.shape[:-1][::-1]):
            axisinfo["Offset"] = int(new_axis_min)
            axisinfo["Size"] = int(axis_shape) - int(new_axis_min)

        self._minindex = new_minindex

    @property
    def dtype(self):
        """
        Property.  The pixel datatype of the remote DVID volume, as a ``numpy.dtype`` object.
        """
        return self._dtype

    @property
    def axiskeys(self):
        """
        Property.  
        A string representing the axis indexing order of the volume, e.g. 'cxyz'
        Always starts with 'c' (channel).
        
        .. note:: By DVID convention, the axiskeys are expressed in fortran order.
        """
        return self._axiskeys
    

    def __init__(self, metadata):
        """
        Constructor.
        
        :param metadata: Either a string containing the json text for the DVID metadata, 
                         or a corresponding dict of metadata (e.g. parsed from the json).
                         If a string is passed, invalid json will result in a ValueError exception.
        """
        assert isinstance( metadata, (dict, str) ), "Expected metadata to be a dict or json str."
        if isinstance( metadata, str ):
            metadata = json.loads( metadata )

        # Check schema...
        #jsonschema.validate( metadata, metadata_schema )

        # Init base class: just copy original metadata
        super( VoxelsMetadata, self ).__init__( **metadata )

        dtypes = []
        for channel_fields in metadata["Properties"]["Values"]:
            dtypes.append( numpy.dtype( channel_fields["DataType"] ) )

        assert all( [dtype == dtypes[0] for dtype in dtypes] ), \
            "Can't support heterogeneous channel types: {}".format( dtypes )
        self._dtype = dtypes[0]
        
        # DVID uses fortran-order notation, so we'll have
        # to reverse the shape before we're done
        shape_fortran = []
        minindex_fortran = []
        minindex_fortran.append( 0 )
        shape_fortran.append( len(metadata["Properties"]["Values"]) ) # channel 
        assert shape_fortran[0] is not None, \
            "Volume metadata is not required to have a complete shape, "\
            "but must at least have a completely specified number of channels."

        for axisfields in metadata['Axes']:
            # If size is 0, then the offset should be ignored.
            if axisfields["Size"] is not None:
                minindex_fortran.append( axisfields["Offset"] )
            else:
                minindex_fortran.append( None )
            if not axisfields["Size"]:
                shape_fortran.append( None )
            else:
                shape_fortran.append( axisfields["Size"] + axisfields["Offset"] )

        # Reverse from F-order to C-order
        self._shape = tuple(reversed(shape_fortran))
        self._minindex = tuple(reversed(minindex_fortran))
    
        axiskeys_fortran = 'c'
        for axisfields in metadata['Axes']:
            axiskeys_fortran += str(axisfields["Label"]).lower()
        
        self._axiskeys = str(axiskeys_fortran[::-1])

    def to_json(self):
        """
        Convenience method: dump this metadata to json string (for transmission to DVID).
        """
        # TODO: Validate schema
        return json.dumps(self)

    @classmethod
    def create_default_metadata(cls, shape, dtype, axiskeys, resolution, units):
        """
        Create a default VoxelsMetadata object from scratch using the given parameters,
        which can then be customized as needed.
        
        Example usage:
        
        .. code-block:: python
        
           metadata = VoxelsMetadata.create_default_metadata( (100,200,300,1), numpy.uint8, 'zyxc', 1.5, "micrometers" )
       
           # Customize: Adjust resolution for Z-axis
           # Note that manual adjustments here must use Fortran-order conventions.
           # Hence, Z is axis 2, not axis 0.
           assert metadata["Axes"][2]["Label"] == "Z"
           metadata["Axes"][2]["Resolution"] = 6.0
   
           # Customize: name channels
           metadata["Properties"]["Values"][0]["Label"] = "intensity-R"
           metadata["Properties"]["Values"][1]["Label"] = "intensity-G"
           metadata["Properties"]["Values"][2]["Label"] = "intensity-B"
   
           # Prepare for transmission: encode to json
           jsontext = metadata.to_json()        
        """
        assert axiskeys == 'zyxc'[-len(axiskeys):], "Axiskeys must be in C-order, and must include the channel axis."
        assert len(axiskeys) == len(shape), "shape/axiskeys mismatch: {} doesn't match {}".format( axiskeys, shape )
        dtype = numpy.dtype(dtype)
        
        metadata = {}
        metadata["Axes"] = []
        metadata["Properties"] = {}
        for key, size in list(zip(axiskeys, shape))[::-1][1:]: # skip channel
            axisdict = {}
            axisdict["Label"] = key.upper()
            axisdict["Resolution"] = resolution
            axisdict["Units"] = units
            axisdict["Size"] = size
            axisdict["Offset"] = 0
            metadata["Axes"].append( axisdict )
        
        metadata["Properties"]["Values"] = []
        num_channels = shape[-1]
        for _ in range( num_channels ):
            metadata["Properties"]["Values"].append( { "DataType" : dtype.name,
                                         "Label" : "" } )
        return VoxelsMetadata(metadata)


    TYPENAMES = { ('uint8',  1) : 'grayscale8',
                  ('uint32', 1) : 'labels32',
                  ('uint64', 1) : 'labels64',
                  ('uint8',  4) : 'rgba8' }

    
    def determine_dvid_typename(self):
        """
        Based on the dtype and number of channels for this volume, 
        determine the datatype name (in DVID terminology).
        For example, if this volume contains 1-channel uint8 data, 
        the DVID datatype is 'grayscale8'.
        """
        # Last axis is always channel
        num_channels = self.shape[-1]
        try:
            return self.TYPENAMES[(self.dtype.name, num_channels)]
        except KeyError:
            msg = "DVID does not have an associated typename for {} channels of pixel type {}"\
                  "".format( num_channels, self.dtype )
            raise Exception( msg )

    @classmethod    
    def determine_channels_from_dvid_typename(cls, typename):
        mapping = { v:k for k,v in list(cls.TYPENAMES.items()) }
        try:
            return mapping[typename]
        except KeyError:
            msg = "Don't support DVID typename '{}'".format( typename )
            raise Exception(msg)
    
    if _have_vigra:
        def create_axistags(self):
            """
            Generate a vigra.AxisTags object corresponding to this VoxelsMetadata.
            (Requires vigra.)
            """
            tags_f = vigra.AxisTags()
            tags_f.insert( 0, vigra.AxisInfo('c', typeFlags=vigra.AxisType.Channels) )
            dtypes = []
            channel_labels = []
            for channel_fields in self["Properties"]["Values"]:
                dtypes.append( numpy.dtype( channel_fields["DataType"] ).type )
                channel_labels.append( channel_fields["Label"] )

            # We monkey-patch the channel labels onto the axistags object as a new member
            tags_f.channelLabels = channel_labels
            for axisfields in self['Axes']:
                key = str(axisfields["Label"]).lower()
                res = axisfields["Resolution"]
                tag = vigra.defaultAxistags(key)[0]
                tag.resolution = res
                tags_f.insert( len(tags_f), tag )
                # TODO: Check resolution units, because apparently 
                #        they can be different from one axis to the next...
    
            assert all( [dtype == dtypes[0] for dtype in dtypes] ), \
                "Can't support heterogeneous channel types: {}".format( dtypes )

            # Reverse from F-order to C-order
            tags_c = vigra.AxisTags(list(tags_f)[::-1])
            return tags_c
        
        @classmethod
        def create_volumeinfo_from_axistags(cls, shape, dtype, axistags):
            assert False, "TODO..."
            

    if _have_h5py:    
        @classmethod
        def create_from_h5_dataset(cls, dataset):
            """
            Create a VolumeInfo object to describe the given h5 dataset object.

            :param dataset: An hdf5 dataset object that meets the following criteria:\n
                            * Indexed in F-order
                            * Has an 'axistags' attribute, produced using vigra.AxisTags.toJSON()
                            * Has an explicit channel axis
                     
            (Requires h5py.)
            """
            dtype = dataset.dtype.type
            shape = dataset.shape
            if 'dvid_metadata' in dataset.attrs:
                metadata_json = dataset.attrs['dvid_metadata']
                metadata = json.loads( metadata_json )
                return VoxelsMetadata( metadata )
            elif _have_vigra and 'axistags' in dataset.attrs:
                axistags = vigra.AxisTags.fromJSON( dataset.attrs['axistags'] )
                return cls.create_volumeinfo_from_axistags( shape, dtype, axistags )
            else:
                # Choose default axiskeys
                default_keys = 'tzyxc'
                axiskeys = default_keys[-len(shape):]
                return VoxelsMetadata.create_default_metadata( shape, dtype, axiskeys, 1.0, "" )

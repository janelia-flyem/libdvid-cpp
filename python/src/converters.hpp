#include <vector>
#include <string>
#include <algorithm>
#include <sstream>

#include <boost/unordered_map.hpp>
#include <boost/assign/list_of.hpp>

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

// We intentionally omit numpy/arrayobject.h here because cpp files need to be careful with exactly when this is imported.
// http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
// Therefore, we assume the cpp file that included converters.hpp has already included numpy/arrayobject.h
// #include <numpy/arrayobject.h>

#include "BinaryData.h"
#include "DVIDException.h"

using namespace boost::python;

namespace libdvid { namespace python {

    //*********************************************************************************************
    //* Helpers
    //*********************************************************************************************

    //! PyBinaryDataHolder is the Python wrapper class for the SmartPtrHolder<BinaryDataPtr> class.
    //! It is exposed here as a global so helper functions can use it to manage BinaryData objects.
    //! It must be assigned in the module init section (see libdvid_python.cpp).
    boost::python::object PyBinaryDataHolder;
    
    template<class SmartPtr>
    struct SmartPtrHolder
    {
        SmartPtrHolder() {}
        SmartPtr ptr;
    };
    typedef SmartPtrHolder<libdvid::BinaryDataPtr> BinaryDataHolder;


    //*********************************************************************************************
    //*********************************************************************************************
    //* Python -> C++
    //*********************************************************************************************
    //*********************************************************************************************

    //!*********************************************************************************************
    //! This struct tells boost::python how to convert Python sequences into std::vector<T>.
    //! To use it, instantiate a single instance of it somewhere in your module init section:
    //!
    //!     libdvid::python::std_vector_from_python_iterable<unsigned int>();
    //!
    //! NOTE: This does NOT convert the other way, i.e. from std::vector<T> into Python list.
    //!      If you need to return a std::vector<T>, you'll have to implement that separately.
    //!
    //! For explanation and examples, see the following links:
    //! https://misspent.wordpress.com/2009/09/27/how-to-write-boost-python-converters
    //! http://www.boost.org/doc/libs/1_39_0/libs/python/doc/v2/faq.html#custom_string
    //!*********************************************************************************************
    template <typename T>
    struct std_vector_from_python_iterable
    {
        typedef std::vector<T> vector_t;
    
        std_vector_from_python_iterable()
        {
          converter::registry::push_back(
              &convertible,
              &construct,
              type_id<vector_t>());
        }
    
        // Determine if obj_ptr can be converted in a vector
        static void* convertible(PyObject* obj_ptr)
        {
            if (!PySequence_Check(obj_ptr))
            {
                return 0;
            }
            return obj_ptr;
        }
    
        // Convert obj_ptr into a vector
        static void construct( PyObject* obj_ptr, converter::rvalue_from_python_stage1_data* data)
        {
            assert( PySequence_Check(obj_ptr) );
    
            // Grab pointer to memory into which to construct the new vector_t
            void* storage = ((converter::rvalue_from_python_storage<vector_t>*) data)->storage.bytes;
    
            // in-place construct the new vector_t
            vector_t * the_vector = new (storage) vector_t;
    
            // Now copy the data from python into the vector
            typedef stl_input_iterator<T> vector_iter_t;
            object obj = object(handle<>(borrowed(obj_ptr)));
            the_vector->assign( vector_iter_t(obj), vector_iter_t()  );
    
            // Stash the memory chunk pointer for later use by boost::python
            data->convertible = storage;
        }
    };

    //!*********************************************************************************************
    //! This converter auto-converts 'None' objects into empty std::strings.
    //!*********************************************************************************************
    struct std_string_from_python_none
    {
        std_string_from_python_none()
        {
          converter::registry::push_back(
              &convertible,
              &construct,
              type_id<std::string>());
        }
    
        // Determine if obj_ptr is None
        static void* convertible(PyObject* obj_ptr)
        {
            if (obj_ptr == Py_None)
            {
                return obj_ptr;
            }
            return 0;
        }
    
        // Convert obj_ptr into a std::string
        static void construct( PyObject* obj_ptr, converter::rvalue_from_python_stage1_data* data)
        {
            assert (obj_ptr == Py_None);
    
            // Grab pointer to memory into which to construct the std::string
            void* storage = ((converter::rvalue_from_python_storage<std::string>*) data)->storage.bytes;
    
            // in-place construct the new std::string
            // extraced from the python object
            new (storage) std::string;
        
            // Stash the memory chunk pointer for later use by boost.python
            data->convertible = storage;
        }
    };

    //!*********************************************************************************************
    //! This helper struct is specialized over integer types to provide a
    //! compile-time mapping from integer types to numpy typenumbers.
    //!*********************************************************************************************
    template <typename T> struct numpy_typenums {};
    template <> struct numpy_typenums<libdvid::uint8> { static const int typenum = NPY_UINT8; };
    template <> struct numpy_typenums<libdvid::uint64> { static const int typenum = NPY_UINT64; };

    //!*********************************************************************************************
    //! Declares a mapping between numpy typenumbers and the corresponding dtype names
    //!*********************************************************************************************
    static boost::unordered_map<int, std::string> dtype_names =
        boost::assign::map_list_of
        (NPY_UINT8, "uint8")
        (NPY_UINT64, "uint64");

    //!*********************************************************************************************
    //! Converts the given numpy ndarray object into a DVIDVoxels object.
    //! NOTE: The data from the ndarray is *copied* into the new DVIDVoxels object.
    //!*********************************************************************************************
    template <class VolumeType>
    struct ndarray_to_volume
    {
        ndarray_to_volume()
        {
          converter::registry::push_back( &convertible, &construct, type_id<VolumeType>() );
        }
    
        static void* convertible(PyObject* obj_ptr)
        {
            if (!PyArray_Check(obj_ptr))
            {
                return 0;
            }
            // We could also check shape, dtype, etc. here, but it's more helpful to 
            // wait until construction time, and then throw a specific exception to
            // tell the user what he did wrong.
            return obj_ptr;
        }

        static void construct( PyObject* obj_ptr, converter::rvalue_from_python_stage1_data* data)
        {
            using namespace boost::python;
            using namespace libdvid;

            assert(PyArray_Check(obj_ptr));
    
            typedef typename VolumeType::voxel_type voxel_type;
    
            object ndarray = object(handle<>(borrowed(obj_ptr)));
            
            // Verify ndarray.dtype
            std::string dtype = extract<std::string>(str(ndarray.attr("dtype")));
            const int numpy_typenum = numpy_typenums<voxel_type>::typenum;
            if (dtype != dtype_names[numpy_typenum])
            {
                std::ostringstream ssMsg;
                ssMsg << "Volume has wrong dtype.  Expected " << dtype_names[numpy_typenum] << ", got " << dtype;
                throw ErrMsg(ssMsg.str());
            }
    
            // Verify ndarray dimensionality.
            int ndarray_ndim = extract<int>(ndarray.attr("ndim"));
            if (ndarray_ndim != VolumeType::num_dims)
            {
                std::string shape = extract<std::string>(str(ndarray.attr("shape")));
                std::ostringstream ssMsg;
                ssMsg << "Volume is not exactly " << VolumeType::num_dims << "D.  Shape is " << shape;
                throw ErrMsg( ssMsg.str() );
            }

            // Verify ndarray memory order
            if (!ndarray.attr("flags")["C_CONTIGUOUS"])
            {
                throw ErrMsg("Volume is not C_CONTIGUOUS");
            }
    
            // Extract dims from ndarray.shape
            object shape = ndarray.attr("shape");
            typedef stl_input_iterator<Dims_t::value_type> shape_iter_t;
            Dims_t dims;
            dims.assign( shape_iter_t(shape), shape_iter_t() );
    
            // Extract the voxel count
            int voxel_count = extract<int>(ndarray.attr("size"));
    
            // Obtain a pointer to the array's data
            PyArrayObject * array_object = reinterpret_cast<PyArrayObject *>( ndarray.ptr() );
            voxel_type const * voxel_data = static_cast<voxel_type const *>( PyArray_DATA(array_object) );

            // Grab pointer to memory into which to construct the DVIDVoxels
            void* storage = ((converter::rvalue_from_python_storage<VolumeType>*) data)->storage.bytes;

            // Create DVIDVoxels<> from ndarray data using "in-place" new().
            // FIXME: The DVIDVoxels constructor copies the data. Is that really necessary?
            VolumeType * volume = new (storage) VolumeType( voxel_data, voxel_count, dims );

            // Stash the memory chunk pointer for later use by boost.python
            data->convertible = storage;
        }
    };

    //!*********************************************************************************************
    //! Convert Python buffer objects (e.g. str, bytearray) -> BinaryDataPtr
    //!*********************************************************************************************
    struct binary_data_ptr_from_python_buffer
    {
        binary_data_ptr_from_python_buffer()
        {
          converter::registry::push_back(
              &convertible,
              &construct,
              type_id<BinaryDataPtr>());
        }

        static void* convertible(PyObject* obj_ptr)
        {
            // The "proper" thing here would be to check if this object is a buffer,
            // but I think it's more helpful to the user if we check inside construct()
            // and raise a descriptive exception if they tried to supply a non-buffer.

            // return PyObject_CheckBuffer(obj_ptr) ? obj_ptr : 0;

            return obj_ptr;
        }

        static void construct( PyObject* obj_ptr, converter::rvalue_from_python_stage1_data* data)
        {
            using namespace libdvid;
            using namespace boost::python;

            BinaryDataPtr binary_data;
            if (obj_ptr == Py_None)
            {
                // As a special convenience, we auto-convert None to an empty buffer.
                binary_data = BinaryData::create_binary_data();
            }
            else
            {
                if (!PyObject_CheckBuffer(obj_ptr))
                {
                    std::string value_str = extract<std::string>(str(object(handle<>(borrowed(obj_ptr)))));
                    throw ErrMsg("Value is not a buffer: " + value_str);
                }
                Py_buffer py_buffer;
                PyObject_GetBuffer(obj_ptr, &py_buffer, PyBUF_SIMPLE);

                // Copy buffer into BinaryData
                binary_data = BinaryData::create_binary_data(static_cast<char*>(py_buffer.buf), py_buffer.len);
                PyBuffer_Release(&py_buffer);
            }

            // Grab pointer to memory into which to construct the BinaryDataPtr
            void* storage = ((converter::rvalue_from_python_storage<BinaryDataPtr>*) data)->storage.bytes;

            // in-place construct the new std::string
            // extraced from the python object
            new (storage) BinaryDataPtr(binary_data);

            // Stash the memory chunk pointer for later use by boost.python
            data->convertible = storage;
        }
    };

    //*********************************************************************************************
    //*********************************************************************************************
    //* C++ -> Python
    //*********************************************************************************************
    //*********************************************************************************************

    //!*********************************************************************************************
    //! Convert BlockXYZ in both directions:
    //! tuple (Python) --> BlockXYZ (C++)
    //! BlockXYZ (C++) --> namedtuple("BlockXYZ", "x y z") (Python)
    //!*********************************************************************************************
    struct block_to_python_block
    {
        block_to_python_block()
        {
            block_to_python_block::register_to_python();
            block_to_python_block::register_from_python();
        }

        static object PyBlockXYZ;
        static void register_to_python()
        {
            object collections = import("collections");
            PyBlockXYZ = collections.attr("namedtuple")("BlockXYZ", "x y z");
            scope().attr("BlockXYZ") = PyBlockXYZ;
            to_python_converter<BlockXYZ, block_to_python_block>();
        }

        static void register_from_python()
        {
            converter::registry::push_back(
                &block_to_python_block::convertible,
                &block_to_python_block::construct,
                type_id<BlockXYZ>());
        }

        // Determine if obj_ptr can be converted to a BlockXYZ
        static void* convertible(PyObject* obj_ptr)
        {
            if (!PySequence_Check(obj_ptr))
            {
                return 0;
            }

            // We could check the length here, but it's nicer to
            // give the user an explanatory exception in construct()
            return obj_ptr;
        }

        // Convert obj_ptr into a BlockXYZ
        static void construct( PyObject* obj_ptr, converter::rvalue_from_python_stage1_data* data)
        {
            assert( PySequence_Check(obj_ptr) );

            // Grab pointer to memory into which to construct the new vector_t
            void* storage = ((converter::rvalue_from_python_storage<BlockXYZ>*) data)->storage.bytes;

            object sequence = object(handle<>(borrowed(obj_ptr)));
            int len = extract<int>(sequence.attr("__len__")());
            if (len != 3)
            {
                std::ostringstream msg;
                msg << "BlockXYZ must have exactly 3 entries, but this sequence contains "
                    << len << " entries.";
                throw ErrMsg(msg.str());
            }

            // in-place construct the new BlockXYZ
            int x = extract<int>(sequence[0]);
            int y = extract<int>(sequence[1]);
            int z = extract<int>(sequence[2]);
            new (storage) BlockXYZ(x, y, z);

            // Stash the memory chunk pointer for later use by boost::python
            data->convertible = storage;
        }


        static PyObject* convert(BlockXYZ const& block)
        {
            return incref(PyBlockXYZ(block.x, block.y, block.z).ptr());
        }
    };
    object block_to_python_block::PyBlockXYZ;

    //!*********************************************************************************************
    //! Convert PointXYZ in both directions:
    //! tuple (Python) --> PointXYZ (C++)
    //! PointXYZ (C++) --> namedtuple("PointXYZ", "x y z") (Python)
    //!*********************************************************************************************
    struct point_to_python_point
    {
    	point_to_python_point()
        {
    		point_to_python_point::register_to_python();
    		point_to_python_point::register_from_python();
        }

        static object PyPointXYZ;
        static void register_to_python()
        {
            object collections = import("collections");
            PyPointXYZ = collections.attr("namedtuple")("PointXYZ", "x y z");
            scope().attr("PointXYZ") = PyPointXYZ;
            to_python_converter<PointXYZ, point_to_python_point>();
        }

        static void register_from_python()
        {
            converter::registry::push_back(
                &point_to_python_point::convertible,
                &point_to_python_point::construct,
                type_id<PointXYZ>());
        }

        // Determine if obj_ptr can be converted to a PointXYZ
        static void* convertible(PyObject* obj_ptr)
        {
            if (!PySequence_Check(obj_ptr))
            {
                return 0;
            }

            // We could check the length here, but it's nicer to
            // give the user an explanatory exception in construct()
            return obj_ptr;
        }

        // Convert obj_ptr into a PointXYZ
        static void construct( PyObject* obj_ptr, converter::rvalue_from_python_stage1_data* data)
        {
            assert( PySequence_Check(obj_ptr) );

            // Grab pointer to memory into which to construct the new vector_t
            void* storage = ((converter::rvalue_from_python_storage<PointXYZ>*) data)->storage.bytes;

            object sequence = object(handle<>(borrowed(obj_ptr)));
            int len = extract<int>(sequence.attr("__len__")());
            if (len != 3)
            {
                std::ostringstream msg;
                msg << "PointXYZ must have exactly 3 entries, but this sequence contains "
                    << len << " entries.";
                throw ErrMsg(msg.str());
            }

            // in-place construct the new PointXYZ
            int x = extract<int>(sequence[0]);
            int y = extract<int>(sequence[1]);
            int z = extract<int>(sequence[2]);
            new (storage) PointXYZ(x, y, z);

            // Stash the memory chunk pointer for later use by boost::python
            data->convertible = storage;
        }


        static PyObject* convert(PointXYZ const& point)
        {
            return incref(PyPointXYZ(point.x, point.y, point.z).ptr());
        }
    };
    object point_to_python_point::PyPointXYZ;


    //!*********************************************************************************************
    //! Convert SubstackXYZ in both directions:
    //! tuple (Python) --> SubstackXYZ (C++)
    //! SubstackXYZ (C++) --> namedtuple("SubstackXYZ", "x y z size") (Python)
    //!*********************************************************************************************
    struct substack_to_python_substack
    {
    	substack_to_python_substack()
        {
    		substack_to_python_substack::register_to_python();
    		substack_to_python_substack::register_from_python();
        }

        static object PySubstackXYZ;
        static void register_to_python()
        {
            object collections = import("collections");
            PySubstackXYZ = collections.attr("namedtuple")("SubstackXYZ", "x y z size");
            scope().attr("SubstackXYZ") = PySubstackXYZ;
            to_python_converter<SubstackXYZ, substack_to_python_substack>();
        }

        static void register_from_python()
        {
            converter::registry::push_back(
                &substack_to_python_substack::convertible,
                &substack_to_python_substack::construct,
                type_id<SubstackXYZ>());
        }

        // Determine if obj_ptr can be converted to a SubstackXYZ
        static void* convertible(PyObject* obj_ptr)
        {
            if (!PySequence_Check(obj_ptr))
            {
                return 0;
            }

            // We could check the length here, but it's nicer to
            // give the user an explanatory exception in construct()
            return obj_ptr;
        }

        // Convert obj_ptr into a SubstackXYZ
        static void construct( PyObject* obj_ptr, converter::rvalue_from_python_stage1_data* data)
        {
            assert( PySequence_Check(obj_ptr) );

            // Grab pointer to memory into which to construct the new vector_t
            void* storage = ((converter::rvalue_from_python_storage<SubstackXYZ>*) data)->storage.bytes;

            object sequence = object(handle<>(borrowed(obj_ptr)));
            int len = extract<int>(sequence.attr("__len__")());
            if (len != 4)
            {
                std::ostringstream msg;
                msg << "SubstackXYZ must have exactly 4 entries, but this sequence contains "
                    << len << " entries.";
                throw ErrMsg(msg.str());
            }

            // in-place construct the new SubstackXYZ
            int x = extract<int>(sequence[0]);
            int y = extract<int>(sequence[1]);
            int z = extract<int>(sequence[2]);
            int size = extract<int>(sequence[3]);
            new (storage) SubstackXYZ(x, y, z, size);

            // Stash the memory chunk pointer for later use by boost::python
            data->convertible = storage;
        }

        static PyObject* convert(SubstackXYZ const& substack)
        {
            return incref(PySubstackXYZ(substack.x, substack.y, substack.z, substack.size).ptr());
        }
    };
    object substack_to_python_substack::PySubstackXYZ;


    //!*********************************************************************************************
    //! This converts BinaryDataPtr objects into Python strings.
    //! NOTE: It copies the data.
    //!*********************************************************************************************
    struct binary_data_ptr_to_python_str
    {
        static PyObject* convert(BinaryDataPtr const& binary_data)
        {
            return PyString_FromStringAndSize( binary_data->get_data().c_str(), binary_data->get_data().size() );
        }
    };
    
    //!*********************************************************************************************
    //! Converts the given DVIDVoxels object into a numpy array.
    //! NOTE:The ndarray will *steal* the data from the DVIDVoxels object.
    //!*********************************************************************************************
    template <class VolumeType> // VolumeType must be some DVIDVoxels<T, N>
    struct volume_to_ndarray
    {
        static PyObject* convert( VolumeType volume )
        {
            using namespace boost::python;
            using namespace libdvid;
    
            typedef typename VolumeType::voxel_type voxel_type;
    
            BinaryDataPtr volume_data = volume.get_binary();
    
            // Wrap the BinaryData in a BinaryDataHolder object,
            // which is managed via Python's reference counting scheme.
            object py_managed_bd = PyBinaryDataHolder();
            BinaryDataHolder & holder = extract<BinaryDataHolder&>(py_managed_bd);
            holder.ptr = volume_data;
    
            // Copy dims to type numpy expects.
            std::vector<npy_intp> numpy_dims( volume.get_dims().begin(), volume.get_dims().end() );
    
            // We will create a new array with the data from the existing PyBinaryDataHolder object (no copy).
            // The basic idea is described in the following link, but can get away with a lot less code
            // because we used boost-python to define the Python type (PyBinaryDataHolder).
            // (Note: this post is old, so the particular API we're using here is slightly different.)
            // http://blog.enthought.com/python/numpy-arrays-with-pre-allocated-memory
            void const * raw_data = static_cast<void const*>(volume_data->get_raw());
            PyObject * array_object = PyArray_SimpleNewFromData( numpy_dims.size(),
                                                                 &numpy_dims[0],
                                                                 numpy_typenums<voxel_type>::typenum,
                                                                 const_cast<void*>(raw_data) );
            if (!array_object)
            {
                throw ErrMsg("Failed to create array from BinaryData!");
            }
            PyArrayObject * ndarray = reinterpret_cast<PyArrayObject *>( array_object );
    
            // As described in the link above, assigning the 'base' pointer here ensures
            //  that the memory is deallocated when the user is done with the ndarray.
            int status = PyArray_SetBaseObject(ndarray, py_managed_bd.ptr());
            if (status != 0)
            {
                throw ErrMsg("Failed to set array base object!");
            }
            // PyArray_SetBaseObject *steals* the reference, so we need to INCREF here
            //  to make sure the binary data object isn't destroyed when we return.
            incref(py_managed_bd.ptr());
    
            // Return the new array.
            return array_object;
        }
    };

    //!*********************************************************************************************
    //! This converts Json::Value objects into python dict objects.
    //! NOTE: This isn't an efficient conversion method.
    //!*********************************************************************************************
    struct json_value_to_dict
    {
        static PyObject* convert(Json::Value const & json_value)
        {
            using namespace boost::python;

            // For now, easiest thing to do is just export as
            //  string and re-parse via python's json module.
            std::ostringstream ss;
            ss << json_value;

            object json = import("json");
            return incref(json.attr("loads")( ss.str() ).ptr());
        }
    };
    
}} // namespace libdvid::python

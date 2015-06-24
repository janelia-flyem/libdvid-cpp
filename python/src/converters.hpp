#include <vector>
#include <string>
#include <sstream>

#include <boost/unordered_map.hpp>
#include <boost/assign/list_of.hpp>
#include <boost/utility/enable_if.hpp>

#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/slice.hpp>

// We intentionally omit numpy/arrayobject.h here because cpp files need to be careful with exactly when this is imported.
// http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
// Therefore, we assume the cpp file that included converters.hpp has already included numpy/arrayobject.h
// #include <numpy/arrayobject.h>

#include "BinaryData.h"
#include "DVIDException.h"
#include "DVIDException.h"

namespace libdvid { namespace python {

	//*********************************************************************************************
	//! The structs in this file implement the functions boost-python needs to convert between
	//!  custom C++ objects and Python objects.
	//!
	//! The functions are actually registered within these structs' constructors, so to activate a
	//!  particular conversion, simply instantiate the appropriate struct in your module
	//!  initialization code.  For example:
	//!
	//!  BOOST_PYTHON_MODULE(_my_module)
	//!  {
	//!     [... import_array(), other initializations, etc. ...]
	//!
	//!     libdvid::python::std_vector_from_python_iterable<unsigned int>();
	//!
	//!     [... function definitions, etc. ...]
	//!  }
	//!
	//! For explanation and examples of the general method, see the following links:
	//! https://misspent.wordpress.com/2009/09/27/how-to-write-boost-python-converters
	//! http://www.boost.org/doc/libs/1_39_0/libs/python/doc/v2/faq.html#custom_string
	//*********************************************************************************************

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
    //!*********************************************************************************************
    template <typename T>
    struct std_vector_from_python_iterable
    {
        typedef std::vector<T> vector_t;
    
        std_vector_from_python_iterable()
        {
            using namespace boost::python;
            converter::registry::push_back( &convertible, &construct, type_id<vector_t>());
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
        static void construct( PyObject* obj_ptr, boost::python::converter::rvalue_from_python_stage1_data* data)
        {
            using namespace boost::python;

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
            using namespace boost::python;
            converter::registry::push_back(&convertible, &construct, type_id<std::string>());
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
        static void construct( PyObject* obj_ptr, boost::python::converter::rvalue_from_python_stage1_data* data)
        {
            using namespace boost::python;

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
    //! Convert Python buffer objects (e.g. str, bytearray) -> BinaryDataPtr
    //!*********************************************************************************************
    struct binary_data_ptr_from_python_buffer
    {
        binary_data_ptr_from_python_buffer()
        {
            using namespace boost::python;
            converter::registry::push_back(&convertible, &construct, type_id<BinaryDataPtr>());
        }

        static void* convertible(PyObject* obj_ptr)
        {
            // The "proper" thing here would be to check if this object is a buffer,
            // but I think it's more helpful to the user if we check inside construct()
            // and raise a descriptive exception if they tried to supply a non-buffer.

            // return PyObject_CheckBuffer(obj_ptr) ? obj_ptr : 0;

            return obj_ptr;
        }

        static void construct( PyObject* obj_ptr, boost::python::converter::rvalue_from_python_stage1_data* data)
        {
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
    //! This converts BinaryDataPtr objects into Python strings.
    //! NOTE: It copies the data.
    //!*********************************************************************************************
    struct binary_data_ptr_to_python_str
    {
    	binary_data_ptr_to_python_str()
    	{
            using namespace boost::python;
            to_python_converter<BinaryDataPtr, binary_data_ptr_to_python_str>();
    	}

    	static PyObject* convert(BinaryDataPtr const& binary_data)
        {
            return PyString_FromStringAndSize( binary_data->get_data().c_str(),
            								   binary_data->get_data().size() );
        }
    };

    //!*********************************************************************************************
    //! This converts Json::Value objects into python dict objects.
    //! NOTE: This isn't an efficient conversion method.
    //!*********************************************************************************************
    struct json_value_to_dict
    {
    	json_value_to_dict()
    	{
            using namespace boost::python;
            to_python_converter<Json::Value, json_value_to_dict>();
    	}

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


    //*********************************************************************************************
    //*********************************************************************************************
    //* C++ <--> Python
    //*********************************************************************************************
    //*********************************************************************************************

    //!*********************************************************************************************
    //! This helper struct is specialized over integer types to provide a
    //! compile-time mapping from integer types to numpy typenumbers.
    //!*********************************************************************************************
    template <typename T> struct numpy_typenums {};
    template <> struct numpy_typenums<uint8> { static const int typenum = NPY_UINT8; };
    template <> struct numpy_typenums<uint64> { static const int typenum = NPY_UINT64; };

    //!*********************************************************************************************
    //! Declares a mapping between numpy typenumbers and the corresponding dtype names
    //!*********************************************************************************************
    static boost::unordered_map<int, std::string> dtype_names =
        boost::assign::map_list_of
        (NPY_UINT8, "uint8")
        (NPY_UINT64, "uint64");

    //!*********************************************************************************************
    //! Converts between numpy ndarray objects DVIDVoxels objects.
    //!*********************************************************************************************
    template <class VolumeType>
    struct ndarray_to_volume
    {
        static boost::python::object PyBinaryDataHolder;

        struct BinaryDataHolder
        {
            BinaryDataPtr ptr;
        };

        ndarray_to_volume()
        {
            using namespace boost::python;

        	// Register python -> C++
        	converter::registry::push_back( &convertible, &construct, type_id<VolumeType>() );

        	// Register C++ -> Python
        	to_python_converter<VolumeType, ndarray_to_volume<VolumeType> >();
        }

        //! Check if the given object is an ndarray
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

        //! Converts the given numpy ndarray object into a DVIDVoxels object.
        //! NOTE: The data from the ndarray is *copied* into the new DVIDVoxels object.
        static void construct( PyObject* obj_ptr, boost::python::converter::rvalue_from_python_stage1_data* data)
        {
            using namespace boost::python;

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
            if (!ndarray.attr("flags")["F_CONTIGUOUS"])
            {
                throw ErrMsg("Volume is not F_CONTIGUOUS");
            }

            // Extract dims from ndarray.shape (which is already in fortran-order)
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

        //! Converts the given DVIDVoxels object into a numpy array.
        //! NOTE:The ndarray will *steal* the data from the DVIDVoxels object.
        static PyObject* convert( VolumeType volume )
        {
            using namespace boost::python;

            typedef typename VolumeType::voxel_type voxel_type;

            BinaryDataPtr volume_data = volume.get_binary();

            // Wrap the BinaryData in a BinaryDataHolder object,
            // which is managed via Python's reference counting scheme.
            object py_managed_bd = PyBinaryDataHolder();
            BinaryDataHolder & holder = extract<BinaryDataHolder&>(py_managed_bd);
            holder.ptr = volume_data;

            // REVERSE from Fortran order (XYZ) to C-order (ZYX)
            // (PyArray_SimpleNewFromData will create in C-order.)
            // Also, copy to type numpy expects
            std::vector<npy_intp> numpy_dims( volume.get_dims().rbegin(),
                                              volume.get_dims().rend() );

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

            // Finally, return a transposed *view* of the array, since the user is expecting fortran order.
            object c_array = object(handle<>(borrowed(array_object)));
            object f_array = c_array.attr("T");

            // Return the new array.
            return incref(f_array.ptr());
        }
    };
    template <class VolumeType>
    boost::python::object ndarray_to_volume<VolumeType>::PyBinaryDataHolder =
		boost::python::class_<ndarray_to_volume<VolumeType>::BinaryDataHolder>("BinaryDataHolder");

    //!*********************************************************************************************
    //! Convert tuple-like types (e.g. BlockXYZ, PointXYZ, SubstackXYZ, Vertex, Edge)
    //! tuple (Python) --> BlockXYZ (C++)
    //! BlockXYZ (C++) --> namedtuple("BlockXYZ", "x y z") (Python)
    //!*********************************************************************************************
	template <class ClassType, typename ElementType, int N>
    struct namedtuple_converter
    {
		typedef const int ClassType::* class_member_ptr;
		typedef std::vector<class_member_ptr> class_member_ptr_vec;

		static std::string classname;
		static class_member_ptr_vec class_member_ptrs;

		namedtuple_converter( std::string const & classname_,
							  std::string const & elementnames,
							  class_member_ptr_vec member_ptrs )
        {
			assert(member_ptrs.size() == N && "Must pass exactly N member pointers for an N-arity type.");
			classname = classname_;
			class_member_ptrs = member_ptrs;
			register_to_python( classname_, elementnames );
			register_from_python();
        }

        static boost::python::object PyTupleLikeClass;
        static void register_to_python( std::string const & classname,
        								std::string const & elementnames )
        {
            using namespace boost::python;
            object collections = import("collections");
            PyTupleLikeClass = collections.attr("namedtuple")(classname, elementnames);
            scope().attr(classname.c_str()) = PyTupleLikeClass;
            to_python_converter<ClassType, namedtuple_converter<ClassType,ElementType,N> >();
        }

        static void register_from_python()
        {
            using namespace boost::python;
            converter::registry::push_back( &convertible, &construct, type_id<ClassType>() );
        }

        // Determine if obj_ptr can be converted to our ClassType
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

        // Convert obj_ptr into a ClassType
        static void construct( PyObject* obj_ptr, boost::python::converter::rvalue_from_python_stage1_data* data)
        {
            using namespace boost::python;

            assert( PySequence_Check(obj_ptr) );

            // Grab pointer to memory into which to construct the new vector_t
            void* storage = ((converter::rvalue_from_python_storage<ClassType>*) data)->storage.bytes;

            object sequence = object(handle<>(borrowed(obj_ptr)));
            int len = extract<int>(sequence.attr("__len__")());
            if (len != N)
            {
                std::ostringstream msg;
                msg << classname << " must have exactly "
					<< N << " entries, but this sequence contains "
                    << len << " entries.";
                throw ErrMsg(msg.str());
            }

            //construct_in_place_from_sequence<ClassType, ElementType, N>::construct(storage, sequence);
            in_place_new_from_sequence<N>(storage, sequence);

            // Stash the memory chunk pointer for later use by boost::python
            data->convertible = storage;
        }

        //!
        //! construction function.
        //! Thanks to enable_if, only one of the following overloads will be active for a given N.
        //!

		template<int NN> // 1
		static void in_place_new_from_sequence( void* storage, boost::python::object sequence,
												typename boost::enable_if_c<NN == 1>::type* = 0 )
	    {
            using namespace boost::python;
	        new (storage) ClassType( extract<ElementType>(sequence[0]) );
	    }

		template<int NN> // 2
		static void in_place_new_from_sequence( void* storage, boost::python::object sequence,
												typename boost::enable_if_c<NN == 2>::type* = 0 )
	    {
            using namespace boost::python;
	        new (storage) ClassType( extract<ElementType>(sequence[0]),
	        						 extract<ElementType>(sequence[1]) );
	    }

		template<int NN> // 3
		static void in_place_new_from_sequence( void* storage, boost::python::object sequence,
												typename boost::enable_if_c<NN == 3>::type* = 0 )
	    {
            using namespace boost::python;
	        new (storage) ClassType( extract<ElementType>(sequence[0]),
	        						 extract<ElementType>(sequence[1]),
	        						 extract<ElementType>(sequence[2]) );
	    }

		template<int NN> // 4
		static void in_place_new_from_sequence( void* storage, boost::python::object sequence,
												typename boost::enable_if_c<NN == 4>::type* = 0 )
	    {
            using namespace boost::python;
	        new (storage) ClassType( extract<ElementType>(sequence[0]),
	        						 extract<ElementType>(sequence[1]),
	        						 extract<ElementType>(sequence[2]),
	        						 extract<ElementType>(sequence[3]) );
	    }

		template<int NN> // 5
		static void in_place_new_from_sequence( void* storage, boost::python::object sequence,
												typename boost::enable_if_c<NN == 5>::type* = 0 )
	    {
            using namespace boost::python;
	        new (storage) ClassType( extract<ElementType>(sequence[0]),
	        						 extract<ElementType>(sequence[1]),
	        						 extract<ElementType>(sequence[2]),
	        						 extract<ElementType>(sequence[3]),
	        						 extract<ElementType>(sequence[4]) );
	    }

		//!
		//! Conversion functions (C++ -> Python)
        //! Thanks to enable_if, only one of the following convert_impl()
		//!  overloads will be active for a given N.
		//!

        static PyObject* convert( ClassType const& block )
        {
        	return convert_impl<N>( block );
        }

		template <int NN> // 1
        static PyObject* convert_impl( ClassType const& block,
        						  	   typename boost::enable_if_c<NN == 1>::type* = 0 )
        {
            using namespace boost::python;
            return incref(PyTupleLikeClass( block.*class_member_ptrs[0] ).ptr());
        }

		template <int NN> // 2
        static PyObject* convert_impl( ClassType const& block,
        						  	   typename boost::enable_if_c<NN == 2>::type* = 0 )
        {
            using namespace boost::python;
            return incref(PyTupleLikeClass( block.*class_member_ptrs[0],
            								block.*class_member_ptrs[1] ).ptr());
        }

		template <int NN> // 3
        static PyObject* convert_impl( ClassType const& block,
        						   	   typename boost::enable_if_c<NN == 3>::type* = 0 )
        {
            using namespace boost::python;
            return incref(PyTupleLikeClass( block.*class_member_ptrs[0],
            								block.*class_member_ptrs[1],
            								block.*class_member_ptrs[2] ).ptr());
        }

		template <int NN> // 4
        static PyObject* convert_impl( ClassType const& block,
        						  	   typename boost::enable_if_c<NN == 4>::type* = 0 )
        {
            using namespace boost::python;
            return incref(PyTupleLikeClass( block.*class_member_ptrs[0],
            								block.*class_member_ptrs[1],
            								block.*class_member_ptrs[2],
            								block.*class_member_ptrs[3] ).ptr());
        }

		template <int NN> // 5
        static PyObject* convert_impl( ClassType const& block,
        						  	   typename boost::enable_if_c<NN == 5>::type* = 0 )
        {
            using namespace boost::python;
            return incref(PyTupleLikeClass( block.*class_member_ptrs[0],
            								block.*class_member_ptrs[1],
            								block.*class_member_ptrs[2],
            								block.*class_member_ptrs[3],
            								block.*class_member_ptrs[4] ).ptr());
        }

    };
	// Linkage for static members
	template <class T, typename E, int N>
    boost::python::object namedtuple_converter<T,E,N>::PyTupleLikeClass;

	template <class T, typename E, int N>
	std::string namedtuple_converter<T,E,N>::classname;

	template <class T, typename E, int N>
	typename namedtuple_converter<T,E,N>::class_member_ptr_vec namedtuple_converter<T,E,N>::class_member_ptrs;

}} // namespace libdvid::python

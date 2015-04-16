#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/unordered_map.hpp>
#include <boost/assign/list_of.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL libdvid_PYTHON_BINDINGS
#include <numpy/arrayobject.h>

#include <sstream>
#include <algorithm>

#include "BinaryData.h"
#include "DVIDConnection.h"
#include "DVIDServerService.h"
#include "DVIDNodeService.h"
#include "DVIDException.h"

//! PyBinaryData is the Python wrapper class defined below for the BinaryData class.
//! It is exposed here as a global so helper functions can instantiate new BinaryData objects.
boost::python::object PyBinaryData;

//! Helper function.
//! Converts the given Json::Value into a Python dict.
boost::python::dict convert_json_to_dict(Json::Value const & json_value)
{
    using namespace boost::python;
    
    // For now, easiest thing to do is just export as 
    //  string and re-parse via python's json module.
    std::ostringstream ss;
    ss << json_value;
    
    object json = import("json");
    return extract<dict>( json.attr("loads")( ss.str() ) );
}

//! Helper function.
//! Converts the given Python object into a BinaryData object.
//! The Python object must support the buffer protocol (e.g. str, bytearray).
libdvid::BinaryDataPtr convert_python_value_to_binary_data(boost::python::object const & value)
{
    using namespace libdvid;
    using namespace boost::python;

    PyObject* py_value = value.ptr();
    if (!PyObject_CheckBuffer(py_value))
    {
        std::string value_str = extract<std::string>(str(value));
        throw ErrMsg("Value is not a buffer: " + value_str);
    }
    Py_buffer py_buffer;
    PyObject_GetBuffer(py_value, &py_buffer, PyBUF_SIMPLE);

    // Copy buffer into BinaryData
    BinaryDataPtr data = BinaryData::create_binary_data(static_cast<char*>(py_buffer.buf), py_buffer.len);
    PyBuffer_Release(&py_buffer);
    return data;
}

//! Python wrapper function for DVIDNodeService::get_typeinfo()
boost::python::dict python_get_typeinfo(libdvid::DVIDNodeService & nodeService, std::string datatype_name)
{
    Json::Value value = nodeService.get_typeinfo(datatype_name);
    return convert_json_to_dict(value);
}

//! Python wrapper function for DVIDNodeService::custom_request()
boost::python::object custom_request( libdvid::DVIDNodeService & nodeService,
                                      std::string endpoint,
                                      boost::python::object payload_object,
                                      libdvid::ConnectionMethod method )
{
    using namespace libdvid;
    using namespace boost::python;

    BinaryDataPtr payload_data;

    // Check for None
    if (payload_object == object())
    {
        payload_data = BinaryData::create_binary_data();
    }
    else
    {
        payload_data = convert_python_value_to_binary_data( payload_object );
    }

    BinaryDataPtr results = nodeService.custom_request(endpoint, payload_data, method);
    PyObject * py_result_body_str = PyString_FromStringAndSize( results->get_data().c_str(), results->get_data().size() );
    return object(handle<>(py_result_body_str));
}

//! Python wrapper function for DVIDNodeService::put()
void put_keyvalue(libdvid::DVIDNodeService & nodeService, std::string keyvalue, std::string key, boost::python::object & value)
{
    using namespace libdvid;

    BinaryDataPtr data;
    try
    {
        data = convert_python_value_to_binary_data(value);
    }
    catch (ErrMsg const & ex)
    {
        throw ErrMsg("Writing to key '" + keyvalue + "/" + key + "' failed: " + ex.what());
    }
    nodeService.put(keyvalue, key, data );
}

//! Python wrapper function for DVIDNodeService::get()
boost::python::object get_keyvalue(libdvid::DVIDNodeService & nodeService, std::string keyvalue, std::string key)
{
    using namespace libdvid;
    using namespace boost::python;

    // Request value
    BinaryDataPtr value = nodeService.get(keyvalue, key);

    // Copy into a python buffer object
    // Is there a way to avoid this copy?
    PyObject * py_str = PyString_FromStringAndSize( value->get_data().c_str(), value->get_data().size() );
    return object(handle<>(py_str));
}

//! Python wrapper function for DVIDNodeService::get_json()
boost::python::dict get_keyvalue_json(libdvid::DVIDNodeService & nodeService, std::string keyvalue, std::string key)
{
    Json::Value value = nodeService.get_json(keyvalue, key);
    return convert_json_to_dict(value);
}

//! Python wrapper function for DVIDConnection::make_request().
//! The payload_object must support the Python buffer protocol (e.g. str, bytearray).
boost::python::tuple make_request( libdvid::DVIDConnection & connection,
                                   std::string endpoint,
                                   libdvid::ConnectionMethod method,
                                   boost::python::object payload_object,
                                   int timeout )
{
    using namespace libdvid;
    using namespace boost::python;

    BinaryDataPtr results = BinaryData::create_binary_data();
    std::string err_msg ;

    BinaryDataPtr payload_data;

    // Check for None
    if (payload_object == object())
    {
        payload_data = BinaryData::create_binary_data();
    }
    else
    {
        payload_data = convert_python_value_to_binary_data( payload_object );
    }

    int status_code = connection.make_request(endpoint, method, payload_data, results, err_msg, DEFAULT, timeout);
    PyObject * py_result_body_str = PyString_FromStringAndSize( results->get_data().c_str(), results->get_data().size() );
    return make_tuple(status_code, object(handle<>(py_result_body_str )), err_msg);
}

//! This helper struct is specialized over integer types to provide a
//! compile-time mapping from integer types to numpy typenumbers.
template <typename T> struct numpy_typenums {};
template <> struct numpy_typenums<libdvid::uint8> { static const int typenum = NPY_UINT8; };
template <> struct numpy_typenums<libdvid::uint64> { static const int typenum = NPY_UINT64; };

//! Declares a mapping between numpy typenumbers and the corresponding dtype names
static boost::unordered_map<int, std::string> dtype_names =
    boost::assign::map_list_of
    (NPY_UINT8, "uint8")
    (NPY_UINT64, "uint64");

/*
 * This struct contains a generic function get_volume() which can be used to
 * retrieve ND data from DVID.  The struct is parameterized on the voxel_type
 * and dimensionality of the volume to be retrieved.
 *
 * Type-specific functions like get_gray3D are implemented below as thin
 * wrappers around get_volume, which does the real work.
 */
template <typename voxel_type, int N>
struct GetVolumeHelper
{
    static const int numpy_typenum = numpy_typenums<voxel_type>::typenum;

    typedef libdvid::DVIDVoxels<voxel_type, N> volume_type;
    typedef volume_type get_volume_fn_signature( std::string datatype_instance,
                                                 libdvid::Dims_t dims,
                                                 std::vector<unsigned int> offset,
                                                 bool throttle,
                                                 bool compress,
                                                 std::string roi );
    typedef boost::function<get_volume_fn_signature> get_volume_fn_t;

    //! Retrieve generic ND data from DVID.
    //! See struct documentation above.
    //!
    //! \param get_volume_fn A function that can retrieve data of type volume_type.
    static boost::python::object get_volume( std::string datatype_instance,
                                             boost::python::object dims_tuple,
                                             boost::python::object offset_tuple,
                                             bool throttle,
                                             bool compress,
                                             boost::python::object roi_str,
                                             get_volume_fn_t const & get_volume_fn )
    {
        using namespace boost::python;
        using namespace libdvid;

        // Extract roi from python string
        std::string roi = "";
        if (roi_str != object())
        {
            roi = extract<std::string>(roi_str);
        }

        // Extract dimensions from python list/tuple
        libdvid::Dims_t dims;
        typedef stl_input_iterator<Dims_t::value_type> dims_iter_t;
        dims.assign( dims_iter_t(dims_tuple), dims_iter_t() );

        // Extract offset from python list/tuple
        typedef stl_input_iterator<unsigned int> offset_iter_t;
        std::vector<unsigned int> offset;
        offset.assign( offset_iter_t(offset_tuple), offset_iter_t() );

        // Create our own BinaryData instance, managed in Python
        object py_managed_bd = PyBinaryData();
        BinaryData & managed_bd = extract<BinaryData&>(py_managed_bd);

        // Retrieve the ND data
        volume_type volume = get_volume_fn( datatype_instance, dims, offset, throttle, compress, roi );
        BinaryDataPtr volume_data = volume.get_binary();

        // Swap the retrieved data into our own (Python-managed) PyBinaryData object.
        managed_bd.get_data().swap( volume_data->get_data() );

        // Prepare the dims array for numpy to read.
        Dims_t volume_dims = volume.get_dims();
        std::vector<npy_intp> numpy_dims;
        std::copy( volume_dims.begin(), volume_dims.end(), std::back_inserter(numpy_dims) );

        // We will create a new array with the data from the existing PyBinaryData object (no copy).
        // The basic idea is described in the following link, but can get away with a lot less code
        // because boost-python already defined the new Python type for us (PyBinaryData).
        // Also, this post is old so the particular API we're using here is slightly different.
        // http://blog.enthought.com/python/numpy-arrays-with-pre-allocated-memory/
        void const * raw_data = static_cast<void const*>(managed_bd.get_raw());
        PyObject * array_object = PyArray_SimpleNewFromData( numpy_dims.size(),
                                                             &numpy_dims[0],
                                                             numpy_typenum,
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
        Py_INCREF(py_managed_bd.ptr());

        // Return the new array.
        return object(handle<>(array_object));
    }
};
template <typename voxel_type, int N>
const int GetVolumeHelper<voxel_type, N>::numpy_typenum;

//! Python wrapper function for DVIDNodeService::get_gray3D()
boost::python::object get_gray3D( libdvid::DVIDNodeService & nodeService,
                                  std::string datatype_instance,
                                  boost::python::object dims_tuple,
                                  boost::python::object offset_tuple,
                                  bool throttle,
                                  bool compress,
                                  boost::python::object roi_str )
{
    using namespace boost::python;
    using namespace libdvid;

    return GetVolumeHelper<uint8, 3>::get_volume(
        datatype_instance,
        dims_tuple,
        offset_tuple,
        throttle,
        compress,
        roi_str,
        boost::bind(&DVIDNodeService::get_gray3D, boost::ref(nodeService), _1, _2, _3, _4, _5, _6) );
}

//! Python wrapper function for DVIDNodeService::get_labels3D()
boost::python::object get_labels3D( libdvid::DVIDNodeService & nodeService,
                                    std::string datatype_instance,
                                    boost::python::object dims_tuple,
                                    boost::python::object offset_tuple,
                                    bool throttle,
                                    bool compress,
                                    boost::python::object roi_str )
{
    using namespace boost::python;
    using namespace libdvid;

    return GetVolumeHelper<uint64, 3>::get_volume(
        datatype_instance,
        dims_tuple,
        offset_tuple,
        throttle,
        compress,
        roi_str,
        boost::bind(&DVIDNodeService::get_labels3D, boost::ref(nodeService), _1, _2, _3, _4, _5, _6) );
}


/*
 * This struct contains a generic function put_volume() which can be used to
 * send ND data to DVID.  The struct is parameterized on the voxel_type and
 * dimensionality of the volume to be sent.
 *
 * Type-specific functions like put_gray3D are implemented below as thin
 * wrappers around put_volume, which does the real work.
 */
template <typename voxel_type, int N>
struct PutVolumeHelper
{
    static const int numpy_typenum = numpy_typenums<voxel_type>::typenum;

    typedef libdvid::DVIDVoxels<voxel_type, N> volume_type;
    typedef void put_volume_fn_signature( std::string datatype_instance,
                                          volume_type & volume,
                                          std::vector<unsigned int> offset,
                                          bool throttle,
                                          bool compress );
    typedef boost::function<put_volume_fn_signature> put_volume_fn_t;

    //! Send generic ND data to DVID.
    //! See struct documentation above.
    //!
    //! \param put_volume_fn A function that can retrieve data of type volume_type.
    static void put_volume( std::string datatype_instance,
                            boost::python::object ndarray,
                            boost::python::object offset_tuple,
                            bool throttle,
                            bool compress,
                            put_volume_fn_t const & put_volume_fn )
    {
        using namespace boost::python;
        using namespace libdvid;

        // Verify ndarray.dtype
        std::string dtype = extract<std::string>(str(ndarray.attr("dtype")));
        if (dtype != dtype_names[numpy_typenum])
        {
            std::ostringstream ssMsg;
            ssMsg << "Volume has wrong dtype.  Expected " << dtype_names[numpy_typenum] << ", got " << dtype;
            throw ErrMsg(ssMsg.str());
        }

        // Verify ndarray dimensionality.
        if (ndarray.attr("ndim") != N)
        {
            std::string shape = extract<std::string>(str(ndarray.attr("shape")));
            std::ostringstream ssMsg;
            ssMsg << "Volume is not exactly " << N << "D.  Shape is " << shape;
            throw ErrMsg( ssMsg.str() );
        }
        // Verify ndarray memory order
        if (!ndarray.attr("flags")["C_CONTIGUOUS"])
        {
            throw ErrMsg("Volume is not C_CONTIGUOUS");
        }

        // Extract dimensions from python list/tuple
        typedef stl_input_iterator<unsigned int> offset_iter_t;
        std::vector<unsigned int> offset;
        offset.assign( offset_iter_t(offset_tuple), offset_iter_t() );

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

        // Create DVIDVoxels<> from ndarray data
        // FIXME: This copies the data.  Is that really necessary?
        volume_type grayscale_vol( voxel_data, voxel_count, dims );

        // Send to DVID
        put_volume_fn( datatype_instance, grayscale_vol, offset, throttle, compress );
    }
};
template <typename voxel_type, int N>
const int PutVolumeHelper<voxel_type, N>::numpy_typenum;

//! Python wrapper function for DVIDNodeService::put_gray3D()
void put_gray3D( libdvid::DVIDNodeService & nodeService,
                 std::string datatype_instance,
                 boost::python::object ndarray,
                 boost::python::object offset_tuple,
                 bool throttle,
                 bool compress)
{
    using namespace boost::python;
    using namespace libdvid;

    PutVolumeHelper<uint8, 3>::put_volume(
        datatype_instance,
        ndarray,
        offset_tuple,
        throttle,
        compress,
        boost::bind(&DVIDNodeService::put_gray3D, boost::ref(nodeService), _1, _2, _3, _4, _5) );
}

//! Python wrapper function for DVIDNodeService::put_labels3D()
void put_labels3D( libdvid::DVIDNodeService & nodeService,
                 std::string datatype_instance,
                 boost::python::object ndarray,
                 boost::python::object offset_tuple,
                 bool throttle,
                 bool compress)
{
    using namespace boost::python;
    using namespace libdvid;

    PutVolumeHelper<uint64, 3>::put_volume(
        datatype_instance,
        ndarray,
        offset_tuple,
        throttle,
        compress,
        boost::bind(&DVIDNodeService::put_labels3D, boost::ref(nodeService), _1, _2, _3, _4, _5, "") );
}

/*
 * Initialize the Python module (_dvid_python)
 * This cpp file should be built as _dvid_python.so
 */
BOOST_PYTHON_MODULE(_dvid_python)
{
    using namespace libdvid;
    using namespace boost::python;

    // http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
    import_array()

    // DVIDConnection python class definition
    class_<DVIDConnection>("DVIDConnection", init<std::string>())
        .def("make_request", &make_request,
                             ( arg("connection"), arg("endpoint"), arg("method"), arg("payload")=object(), arg("timeout")=DVIDConnection::DEFAULT_TIMEOUT ))
        .def("get_addr", &DVIDConnection::get_addr)
        .def("get_uri_root", &DVIDConnection::get_uri_root)
    ;

    enum_<ConnectionMethod>("ConnectionMethod")
        .value("GET", GET)
        .value("POST", POST)
        .value("PUT", PUT)
        .value("DELETE", DELETE)
    ;

    // DVIDServerService python class definition
    class_<DVIDServerService>("DVIDServerService", init<std::string>())
        .def("create_new_repo", &DVIDServerService::create_new_repo)
    ;

    // DVIDNodeService python class definition
    class_<DVIDNodeService>("DVIDNodeService", init<std::string, UUID>())
        .def("get_typeinfo", &python_get_typeinfo)
        .def("create_graph", &DVIDNodeService::create_graph)
        .def("custom_request", &custom_request)
        
        // keyvalue
        .def("create_keyvalue", &DVIDNodeService::create_keyvalue)
        .def("put", &put_keyvalue)
        .def("get", &get_keyvalue)
        .def("get_json", &get_keyvalue_json)

        // grayscale
        .def("create_grayscale8", &DVIDNodeService::create_grayscale8)
        .def("get_gray3D", &get_gray3D,
            ( arg("service"), arg("instance"), arg("dims"), arg("offset"), arg("throttle")=true, arg("compress")=false, arg("roi")=object()))
        .def("put_gray3D", &put_gray3D,
            ( arg("service"), arg("instance"), arg("ndarray"), arg("offset"), arg("throttle")=true, arg("compress")=false))

        // labels
       .def("create_labelblk", &DVIDNodeService::create_labelblk)
       .def("get_labels3D", &get_labels3D,
            ( arg("service"), arg("instance"), arg("dims"), arg("offset"), arg("throttle")=true, arg("compress")=false, arg("roi")=object()))
       .def("put_labels3D", &put_labels3D,
            ( arg("service"), arg("instance"), arg("ndarray"), arg("offset"), arg("throttle")=true, arg("compress")=false))
    ;

    // Define a python version of BinaryData, and keep a global
    //  reference to it so we can instantiate it ourselves.
    PyBinaryData = class_<BinaryData, BinaryDataPtr>("BinaryData", no_init)
        .def("__init__", make_constructor<BinaryDataPtr()>(&BinaryData::create_binary_data))
    ;

}

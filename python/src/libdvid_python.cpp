#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL libdvid_PYTHON_BINDINGS
#include <numpy/arrayobject.h>

#include <sstream>
#include <algorithm>

#include "BinaryData.h"
#include "DVIDConnection.h"
#include "DVIDNodeService.h"
#include "DVIDException.h"

// PyBinaryData is the Python wrapper class defined below for the BinaryData class.
// It is exposed here as a global so helper functions can instantiate new BinaryData objects.
boost::python::object PyBinaryData;

// Compile-time mapping from integer types to numpy typenumbers
template <typename T> struct numpy_typenums {};
template <> struct numpy_typenums<libdvid::uint8> { static const int typenum = NPY_UINT8; };
template <> struct numpy_typenums<libdvid::uint64> { static const int typenum = NPY_UINT64; };

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

boost::python::dict python_get_typeinfo(libdvid::DVIDNodeService & nodeService, std::string datatype_name)
{
    Json::Value value = nodeService.get_typeinfo(datatype_name);
    return convert_json_to_dict(value);
}

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

boost::python::dict get_keyvalue_json(libdvid::DVIDNodeService & nodeService, std::string keyvalue, std::string key)
{
    Json::Value value = nodeService.get_json(keyvalue, key);
    return convert_json_to_dict(value);
}

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

    std::string roi = "";
    if (roi_str != object())
    {
        roi = extract<std::string>(roi_str);
    }

    libdvid::Dims_t dims;
    typedef stl_input_iterator<Dims_t::value_type> dims_iter_t;
    dims.assign( dims_iter_t(dims_tuple), dims_iter_t() );

    typedef stl_input_iterator<unsigned int> offset_iter_t;
    std::vector<unsigned int> offset;
    offset.assign( offset_iter_t(offset_tuple), offset_iter_t() );

    // Create our own BinaryData instance, managed in Python
    object py_managed_bd = PyBinaryData();
    BinaryData & managed_bd = extract<BinaryData&>(py_managed_bd);

    Grayscale3D volume = nodeService.get_gray3D( datatype_instance, dims, offset, throttle, compress, roi );
    BinaryDataPtr volume_data = volume.get_binary();

    // Swap the data into our managed object.
    managed_bd.get_data().swap( volume_data->get_data() );

    // Create a new array with the data from the existing PyBinaryData object (no copy).
    // The basic idea is described in the following link, but can get away with a lot less code
    // because boost-python already defined the new Python type for us (PyBinaryData).
    // Also, this post is old so the particular API we're using here is slightly different.
    // http://blog.enthought.com/python/numpy-arrays-with-pre-allocated-memory/
    Dims_t volume_dims = volume.get_dims();
    std::vector<npy_intp> numpy_dims;
    std::copy( volume_dims.begin(), volume_dims.end(), std::back_inserter(numpy_dims) );
    void const * raw_data = static_cast<void const*>(managed_bd.get_raw());

    PyObject * array_object = PyArray_SimpleNewFromData( numpy_dims.size(),
                                                         &numpy_dims[0],
                                                         numpy_typenums<uint8>::typenum,
                                                         const_cast<void*>(raw_data) );
    if (!array_object)
    {
        throw ErrMsg("Failed to create array from BinaryData!");
    }
    Py_INCREF(py_managed_bd.ptr());
    PyArrayObject * ndarray = reinterpret_cast<PyArrayObject *>( array_object );

    // As described in the link above, assigning the 'base' pointer here ensures
    //  that the memory is deallocated when the user is done with the ndarray.
    int status = PyArray_SetBaseObject(ndarray, py_managed_bd.ptr());
    if (status != 0)
    {
        throw ErrMsg("Failed to set array base object!");
    }
    return object(handle<>(array_object));
}

void put_gray3D( libdvid::DVIDNodeService & nodeService,
                 std::string datatype_instance,
                 boost::python::object ndarray,
                 boost::python::object offset_tuple,
                 bool throttle=true,
                 bool compress=false )
{
    using namespace boost::python;
    using namespace libdvid;

    typedef stl_input_iterator<unsigned int> offset_iter_t;
    std::vector<unsigned int> offset;
    offset.assign( offset_iter_t(offset_tuple), offset_iter_t() );

    if (str(ndarray.attr("dtype")) != "uint8")
    {
        std::string dtype = extract<std::string>(str(ndarray.attr("dtype")));
        throw ErrMsg("Volume has wrong dtype.  Expected uint8, got " + dtype);
    }
    if (ndarray.attr("ndim") != 3)
    {
        std::string shape = extract<std::string>(str(ndarray.attr("shape")));
        throw ErrMsg("Volume is not exactly 3D.  Shape is " + shape);
    }
    if (!ndarray.attr("flags")["C_CONTIGUOUS"])
    {
        throw ErrMsg("Volume is not C_CONTIGUOUS");
    }

    object shape = ndarray.attr("shape");
    typedef stl_input_iterator<Dims_t::value_type> shape_iter_t;
    Dims_t dims;
    dims.assign( shape_iter_t(shape), shape_iter_t() );

    PyArrayObject * array_object = reinterpret_cast<PyArrayObject *>( ndarray.ptr() );
    Grayscale3D grayscale_vol( static_cast<uint8 const *>( PyArray_DATA(array_object) ),
                               extract<int>(ndarray.attr("size")) * sizeof(Grayscale3D::voxel_type),
                               dims );
    nodeService.put_gray3D( datatype_instance, grayscale_vol, offset, throttle, compress );
}


BOOST_PYTHON_MODULE(_dvid_python)
{
    using namespace libdvid;
    using namespace boost::python;

    // http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
    import_array()

    enum_<ConnectionMethod>("ConnectionMethod")
        .value("GET", GET)
        .value("POST", POST)
        .value("PUT", PUT)
        .value("DELETE", DELETE)
    ;

    class_<DVIDConnection>("DVIDConnection", init<std::string>())
        .def("make_request", &make_request,
                             ( arg("connection"), arg("endpoint"), arg("method"), arg("payload")=object(), arg("timeout")=DVIDConnection::DEFAULT_TIMEOUT ))
        .def("get_addr", &DVIDConnection::get_addr)
        .def("get_uri_root", &DVIDConnection::get_uri_root)
    ;

    PyBinaryData = class_<BinaryData, BinaryDataPtr>("BinaryData", no_init)
        .def("__init__", make_constructor<BinaryDataPtr()>(&BinaryData::create_binary_data))
    ;

    class_<DVIDNodeService>("DVIDNodeService", init<std::string, UUID>())
        .def("get_typeinfo", &python_get_typeinfo)
        .def("create_labelblk", &DVIDNodeService::create_labelblk)
        .def("create_graph", &DVIDNodeService::create_graph)
        
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
    ;

}

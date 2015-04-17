#include <Python.h>
#include <boost/python.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL libdvid_PYTHON_BINDINGS
#include <numpy/arrayobject.h>

#include <sstream>

#include "BinaryData.h"
#include "DVIDConnection.h"
#include "DVIDServerService.h"
#include "DVIDNodeService.h"
#include "DVIDException.h"

#include "converters.hpp"

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

    // Register custom Python -> C++ converters.
    libdvid::python::std_vector_from_python_iterable<unsigned int>();
    libdvid::python::std_string_from_python_none(); // None -> std::string("")

    libdvid::python::ndarray_to_volume<Grayscale3D>();
    libdvid::python::ndarray_to_volume<Labels3D>();
    libdvid::python::ndarray_to_volume<Grayscale2D>();
    libdvid::python::ndarray_to_volume<Labels2D>();

    // Register custom C++ -> Python converters.
    to_python_converter<BinaryDataPtr, libdvid::python::binary_data_ptr_to_python_str>();
    to_python_converter<Json::Value, libdvid::python::json_value_to_dict>();

    to_python_converter<Grayscale3D, libdvid::python::volume_to_ndarray<Grayscale3D> >();
    to_python_converter<Labels3D, libdvid::python::volume_to_ndarray<Labels3D> >();
    to_python_converter<Grayscale2D, libdvid::python::volume_to_ndarray<Grayscale2D> >();
    to_python_converter<Labels2D, libdvid::python::volume_to_ndarray<Labels2D> >();

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

    // For overloaded functions, boost::python needs help figuring out which one we're aiming for.
    // These function pointers specify the ones we want.
    Grayscale3D (DVIDNodeService::*get_gray3D)(std::string, Dims_t, std::vector<unsigned int>, bool, bool, std::string) = &DVIDNodeService::get_gray3D;
    Labels3D (DVIDNodeService::*get_labels3D)(std::string, Dims_t, std::vector<unsigned int>, bool, bool, std::string) = &DVIDNodeService::get_labels3D;
    void (DVIDNodeService::*put_gray3D)(std::string, Grayscale3D const&, std::vector<unsigned int>, bool, bool) = &DVIDNodeService::put_gray3D;
    void (DVIDNodeService::*put_labels3D)(std::string, Labels3D const&, std::vector<unsigned int>, bool, bool, std::string) = &DVIDNodeService::put_labels3D;

    // DVIDNodeService python class definition
    class_<DVIDNodeService>("DVIDNodeService", init<std::string, UUID>())
        .def("get_typeinfo", &DVIDNodeService::get_typeinfo)
        .def("create_graph", &DVIDNodeService::create_graph)
        .def("custom_request", &custom_request)
        
        // keyvalue
        .def("create_keyvalue", &DVIDNodeService::create_keyvalue)
        .def("put", &put_keyvalue)
        .def("get", &DVIDNodeService::get)
        .def("get_json", &DVIDNodeService::get_json)

        // grayscale
        .def("create_grayscale8", &DVIDNodeService::create_grayscale8)
        .def("get_gray3D", get_gray3D,
            ( arg("service"), arg("instance"), arg("dims"), arg("offset"), arg("throttle")=true, arg("compress")=false, arg("roi")=object() ))
        .def("put_gray3D", put_gray3D,
            ( arg("service"), arg("instance"), arg("ndarray"), arg("offset"), arg("throttle")=true, arg("compress")=false))

        // labels
       .def("create_labelblk", &DVIDNodeService::create_labelblk)
       .def("get_labels3D", get_labels3D,
            ( arg("service"), arg("instance"), arg("dims"), arg("offset"), arg("throttle")=true, arg("compress")=false, arg("roi")=object() ))
       .def("put_labels3D", put_labels3D,
            ( arg("service"), arg("instance"), arg("ndarray"), arg("offset"), arg("throttle")=true, arg("compress")=false, arg("roi")=object() ))

        // 2D slices
        .def("get_tile_slice", &DVIDNodeService::get_tile_slice)
        .def("get_tile_slice_binary", &DVIDNodeService::get_tile_slice_binary)
    ;

    enum_<Slice2D>("Slice2D")
        .value("XY", XY)
        .value("XZ", XZ)
        .value("YZ", YZ)
    ;

    // Define a python version of BinaryDataHolder, and keep a global
    //  reference to it so we can instantiate it ourselves.
    libdvid::python::PyBinaryDataHolder = class_<libdvid::python::BinaryDataHolder>("BinaryDataHolder");
}

#include <Python.h>
#include <boost/python.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL libdvid_PYTHON_BINDINGS
#include <numpy/arrayobject.h>

#include <sstream>

#include "DVIDConnection.h"
#include "DVIDNodeService.h"
#include "DVIDException.h"

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
                            ( arg("connection"), arg("endpoint"), arg("method"), arg("payload")=str(), arg("timeout")=DVIDConnection::DEFAULT_TIMEOUT ))
        .def("get_addr", &DVIDConnection::get_addr)
        .def("get_uri_root", &DVIDConnection::get_uri_root)
    ;

    class_<DVIDNodeService>("DVIDNodeService", init<std::string, UUID>())
        .def("get_typeinfo", &python_get_typeinfo)
        .def("create_grayscale8", &DVIDNodeService::create_grayscale8)
        .def("create_labelblk", &DVIDNodeService::create_labelblk)
        .def("create_graph", &DVIDNodeService::create_graph)
        
        // keyvalue
        .def("create_keyvalue", &DVIDNodeService::create_keyvalue)
        .def("put", &put_keyvalue)
        .def("get", &get_keyvalue)
        .def("get_json", &get_keyvalue_json)
    ;

}

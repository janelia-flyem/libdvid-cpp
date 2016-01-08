#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/foreach.hpp>
#include <boost/assign/list_of.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL libdvid_PYTHON_BINDINGS
#include <numpy/arrayobject.h>

#include "BinaryData.h"
#include "DVIDConnection.h"
#include "DVIDServerService.h"
#include "DVIDNodeService.h"
#include "DVIDException.h"

#include "converters.hpp"

namespace libdvid { namespace python {

    //! Python wrapper function for DVIDConnection::make_request().
    //! (Since "return-by-reference" is not an option in Python, boost::python can't provide an automatic wrapper.)
    //! Returns a tuple: (status, result_body, error_msg)
    boost::python::tuple make_request( DVIDConnection & connection,
                                       std::string endpoint,
                                       ConnectionMethod method,
                                       BinaryDataPtr payload_data,
                                       int timeout )
    {
        using namespace boost::python;

        BinaryDataPtr results = BinaryData::create_binary_data();
        std::string err_msg ;

        int status_code = connection.make_request(endpoint, method, payload_data, results, err_msg, DEFAULT, timeout);
        return make_tuple(status_code, object(results), err_msg);
    }

    //! Python wrapper function for DVIDNodeService::get_roi().
    //! Instead of requiring the user to pass an "out-parameter" (not idiomatic in python),
    //! This wrapper function returns the result as a python list of BlockXYZ objects.
    boost::python::list get_roi( DVIDNodeService & nodeService, std::string roi_name )
    {
    	using namespace boost::python;

    	// Retrieve from DVID
    	std::vector<BlockXYZ> result_vector;
    	nodeService.get_roi( roi_name, result_vector );

    	// Convert to Python list
    	list result_list;
    	BOOST_FOREACH(BlockXYZ const & block, result_vector)
    	{
    		result_list.append( static_cast<object>(block) );
    	}
    	return result_list;
    }

    //! Python wrapper function for DVIDConnection::get_roi_partition().
    //! (Since "return-by-reference" is not an option in Python, boost::python can't provide an automatic wrapper.)
    //! Returns a tuple: (status, result_body, error_msg)
    boost::python::tuple get_roi_partition( DVIDNodeService & nodeService,
    										std::string roi_name,
											unsigned int partition_size )
    {
        using namespace boost::python;

        // Retrieve from DVID
    	std::vector<SubstackXYZ> result_substacks;
    	double packing_factor = nodeService.get_roi_partition( roi_name, result_substacks, partition_size );

    	// Convert to Python list
    	list result_list;
    	BOOST_FOREACH(SubstackXYZ const & substack, result_substacks)
    	{
    		result_list.append( static_cast<object>(substack) );
    	}
    	return make_tuple(result_list, packing_factor);
    }

    //! Python wrapper function for DVIDNodeService::roi_ptquery().
    //! Instead of requiring the user to pass an "out-parameter" (not idiomatic in python),
    //! This wrapper function returns the result as a python list of bools.
    boost::python::list roi_ptquery( DVIDNodeService & nodeService,
    				  	  	  	     std::string roi_name,
									 const std::vector<PointXYZ>& points )
    {
    	using namespace boost::python;

    	// Retrieve from DVID
    	std::vector<bool> result_vector;
    	nodeService.roi_ptquery( roi_name, points, result_vector );

    	// Convert to Python list
    	list result_list;
    	BOOST_FOREACH(bool b, result_vector)
    	{
    		result_list.append( static_cast<object>(b) );
    	}
    	return result_list;
    }


    //! Create a new python Exception type.
    //! Copied from: http://stackoverflow.com/a/9690436/162094
    //! \param name The exception name
    //! \param baseTypeObj The base class of this new exception type (default: Exception)
    PyObject* create_exception_class(const char* name, PyObject* baseTypeObj = PyExc_Exception)
    {
        using std::string;
        namespace bp = boost::python;

        string scopeName = bp::extract<string>(bp::scope().attr("__name__"));
        string qualifiedName0 = scopeName + "." + name;
        char* qualifiedName1 = const_cast<char*>(qualifiedName0.c_str());

        PyObject* typeObj = PyErr_NewException(qualifiedName1, baseTypeObj, 0);
        if(!typeObj) bp::throw_error_already_set();
        bp::scope().attr(name) = bp::handle<>(bp::borrowed(typeObj));
        return typeObj;
    }

    // These are the python objects representing Python exception types in libdvid.
    // They are initialized in the module definition (below).
    PyObject * PyErrMsg;
    PyObject * PyDVIDException; // Initialized in module definition (below)

	//! Translator function for libdvid::ErrMsg
    //!
    void translate_ErrMsg(ErrMsg const & e)
    {
        PyErr_SetString(PyErrMsg, e.what());
    }

	//! Translator function for libdvid::DVIDException
    //!
    void translate_DVIDException(DVIDException const & e)
    {
        using namespace boost::python;
        PyErr_SetObject(PyDVIDException, make_tuple(e.getStatus(), e.what()).ptr());
    }

    /*
     * Initialize the Python module (_dvid_python)
     * This cpp file should be built as _dvid_python.so
     */
    BOOST_PYTHON_MODULE(_dvid_python)
    {
        using namespace boost::python;

        // http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
        import_array();

        // Create custom Python exception types for the C++ exceptions defined in DVIDException.h,
        // and register a translator for each
        PyErrMsg = create_exception_class("ErrMsg");
        register_exception_translator<ErrMsg>(&translate_ErrMsg);

        PyDVIDException = create_exception_class("DVIDException", PyErrMsg);
        register_exception_translator<DVIDException>(&translate_DVIDException);

        // Register custom Python -> C++ converters.
        std_vector_from_python_iterable<int>();
        std_vector_from_python_iterable<unsigned int>();
        std_vector_from_python_iterable<BlockXYZ>();
        std_vector_from_python_iterable<SubstackXYZ>();
        std_vector_from_python_iterable<PointXYZ>();
        std_vector_from_python_iterable<Vertex>();
        std_vector_from_python_iterable<Edge>();

        ndarray_to_volume<Grayscale3D>();
        ndarray_to_volume<Labels3D>();
        ndarray_to_volume<Grayscale2D>();
        ndarray_to_volume<Labels2D>();

        // BlockXYZ
        namedtuple_converter<BlockXYZ, int, 3>::class_member_ptr_vec block_members =
			boost::assign::list_of(&BlockXYZ::x)(&BlockXYZ::y)(&BlockXYZ::z);
        namedtuple_converter<BlockXYZ, int, 3>("BlockXYZ", "x y z", block_members);


        /*
        // Vertex 
        namedtuple_converter<Vertex, int, 2>::class_member_ptr_vec vertex_members =
			boost::assign::list_of(&Vertex::id)(&Vertex::weight);
        namedtuple_converter<Vertex, int, 2>("Vertex", "id weight", vertex_members);


        // Edge 
        namedtuple_converter<Edge, int, 3>::class_member_ptr_vec edge_members =
			boost::assign::list_of(&Edge::id1)(&Edge::id2)(&Edge::weight);
        namedtuple_converter<Edge, int, 3>("Edge", "x y z", edge_members);
        */

        // PointXYZ
        namedtuple_converter<PointXYZ, int, 3>::class_member_ptr_vec point_members =
			boost::assign::list_of(&PointXYZ::x)(&PointXYZ::y)(&PointXYZ::z);
        namedtuple_converter<PointXYZ, int, 3>("PointXYZ", "x y z", point_members);

        // SubstackXYZ
        namedtuple_converter<SubstackXYZ, int, 4>::class_member_ptr_vec substack_members =
			boost::assign::list_of(&SubstackXYZ::x)(&SubstackXYZ::y)(&SubstackXYZ::z)(&SubstackXYZ::size);
        namedtuple_converter<SubstackXYZ, int, 4>("SubstackXYZ", "x y z size", substack_members);

        binary_data_ptr_to_python_str();
        binary_data_ptr_from_python_buffer();
        json_value_to_dict();
        std_string_from_python_none(); // None -> std::string("")

        // This special type wraps vector<string> and makes it accessible to Python when used as a return value.
        // For example, see DVIDNodeService::get_keys()
        typedef std::vector<std::string> StringVec;
        class_<StringVec>("StringVec")
                .def(vector_indexing_suite<StringVec>() );

        // DVIDConnection python class definition
        class_<DVIDConnection>("DVIDConnection", init<std::string>())
            .def("make_head_request", &DVIDConnection::make_head_request)
            .def("make_request", &make_request,
                                 ( arg("connection"), arg("endpoint"), arg("method"), arg("payload")=object(), arg("timeout")=DVIDConnection::DEFAULT_TIMEOUT ))
            .def("get_addr", &DVIDConnection::get_addr)
            .def("get_uri_root", &DVIDConnection::get_uri_root)
        ;

        enum_<ConnectionMethod>("ConnectionMethod")
            .value("HEAD", HEAD)
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
        void          (DVIDNodeService::*put_binary)(std::string, std::string, BinaryDataPtr)                                   = &DVIDNodeService::put;
        Grayscale3D   (DVIDNodeService::*get_gray3D)(std::string, Dims_t, std::vector<int>, bool, bool, std::string)            = &DVIDNodeService::get_gray3D;
        Roi3D         (DVIDNodeService::*get_roi3D)(std::string, Dims_t, std::vector<int>, bool, bool)                          = &DVIDNodeService::get_roi3D;
        Labels3D      (DVIDNodeService::*get_labels3D)(std::string, Dims_t, std::vector<int>, bool, bool, std::string)          = &DVIDNodeService::get_labels3D;
        void          (DVIDNodeService::*put_gray3D)(std::string, Grayscale3D const&, std::vector<int>, bool, bool)             = &DVIDNodeService::put_gray3D;
        void          (DVIDNodeService::*put_labels3D)(std::string, Labels3D const&, std::vector<int>, bool, bool, std::string, bool) = &DVIDNodeService::put_labels3D;
        bool          (DVIDNodeService::*create_labelblk)(std::string, std::string)                                             = &DVIDNodeService::create_labelblk;
        BinaryDataPtr (DVIDNodeService::*custom_request)(std::string, BinaryDataPtr, ConnectionMethod, bool)                    = &DVIDNodeService::custom_request;

        // DVIDNodeService python class definition
        class_<DVIDNodeService>("DVIDNodeService", init<std::string, UUID>())
            .def("get_typeinfo", &DVIDNodeService::get_typeinfo)
            .def("create_graph", &DVIDNodeService::create_graph)
            .def("custom_request", custom_request, (arg("endpoint"), arg("payload"), arg("method"), arg("compress")=false))

            // keyvalue
            .def("create_keyvalue", &DVIDNodeService::create_keyvalue)
            .def("put", put_binary)
            .def("get", &DVIDNodeService::get)
            .def("get_json", &DVIDNodeService::get_json)
            .def("get_keys", &DVIDNodeService::get_keys)

            // grayscale
            .def("create_grayscale8", &DVIDNodeService::create_grayscale8)
            .def("get_gray3D", get_gray3D,
                ( arg("service"), arg("instance"), arg("dims"), arg("offset"), arg("throttle")=true, arg("compress")=false, arg("roi")=object() ))
            .def("put_gray3D", put_gray3D,
                ( arg("service"), arg("instance"), arg("ndarray"), arg("offset"), arg("throttle")=true, arg("compress")=false))

            // labels
            .def("create_labelblk", create_labelblk, (arg("service"), arg("instance"), arg("instance2")=object() ))
            .def("get_labels3D", get_labels3D,
                ( arg("service"), arg("instance"), arg("dims"), arg("offset"), arg("throttle")=true, arg("compress")=false, arg("roi")=object() ))
            .def("get_label_by_location",  &DVIDNodeService::get_label_by_location)
            .def("put_labels3D", put_labels3D,
                ( arg("service"), arg("instance"), arg("ndarray"), arg("offset"), arg("throttle")=true, arg("compress")=false, arg("roi")=object(), arg("mutate")=false ))
            .def("body_exists", &DVIDNodeService::body_exists)

            // 2D slices
            .def("get_tile_slice", &DVIDNodeService::get_tile_slice)
            .def("get_tile_slice_binary", &DVIDNodeService::get_tile_slice_binary)


            // graph
            .def("update_vertices", &DVIDNodeService::update_vertices)
            .def("update_edges", &DVIDNodeService::update_edges)

            // ROI
            .def("create_roi", &DVIDNodeService::create_roi)
            .def("get_roi", &get_roi)
            .def("post_roi", &DVIDNodeService::post_roi)
            .def("get_roi_partition", &get_roi_partition)
            .def("roi_ptquery", &roi_ptquery)
            .def("get_roi3D", get_roi3D,
                ( arg("service"), arg("instance"), arg("dims"), arg("offset"), arg("throttle")=true, arg("compress")=false ))			;


        class_<Vertex>("Vertex", init<VertexID, double>());
        class_<Edge>("Edge", init<VertexID, VertexID, double>());

        enum_<Slice2D>("Slice2D")
            .value("XY", XY)
            .value("XZ", XZ)
            .value("YZ", YZ)
        ;

    }

}} // namespace libdvid::python

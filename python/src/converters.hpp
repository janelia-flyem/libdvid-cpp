#include <vector>
#include <string>
#include <boost/python.hpp>

using namespace boost::python;

namespace libdvid { namespace python {

    /*
     * This struct tells boost::python how to convert Python sequences into std::vector<T>.
     * To use it, instantiate a single instance of it somewhere in your module init section:
     * 
     *     libdvid::python::std_vector_from_python_iterable<unsigned int>();
     * 
     * NOTE: This does NOT convert the other way, i.e. from std::vector<T> into Python list.
     *       If you need to return a std::vector<T>, you'll have to implement that separately.
     * 
     * For explanation and examples, see the following links:
     * https://misspent.wordpress.com/2009/09/27/how-to-write-boost-python-converters
     * http://www.boost.org/doc/libs/1_39_0/libs/python/doc/v2/faq.html#custom_string
     */
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
            object obj = object(handle<>(obj_ptr));
            the_vector->assign( vector_iter_t(obj), vector_iter_t()  );
    
            // Stash the memory chunk pointer for later use by boost::python
            data->convertible = storage;
        }
    };

    /*
     * This converter auto-converts 'None' objects into empty std::strings.
     */
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
}}
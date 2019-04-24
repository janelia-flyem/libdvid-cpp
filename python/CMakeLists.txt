cmake_minimum_required (VERSION 2.8) 

######################################################################
#
#      find Python interpreter and libraries
#
######################################################################

find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)

######################################################################
#
#      find default install directory for Python modules
#      (usually PYTHONDIR/Lib/site-packages)
#
######################################################################
SET(LIBDVID_PYTHON_INSTALL_DIR ${Python_SITEARCH}
    CACHE PATH "where to install the libdvid Python package" FORCE)
# this is the install path relative to CMAKE_INSTALL_PREFIX,
# use this in INSTALL() commands to get packaging right
FILE(RELATIVE_PATH LIBDVID_PYTHON_INSTALL_DIR ${CMAKE_INSTALL_PREFIX} ${LIBDVID_PYTHON_INSTALL_DIR})

######################################################################
#
#      find numpy include directory
#      (usually below PYTHONDIR/Lib/site-packages/numpy)
#
######################################################################
SET(PYTHON_NUMPY_INCLUDE_DIR ${Python_NumPy_INCLUDE_DIRS}
    CACHE PATH "Path to numpy include files" FORCE)

######################################################################
#
#      get numpy version
#
######################################################################
IF(NOT PYTHON_NUMPY_NOT_FOUND)
    execute_process(COMMAND "${Python_EXECUTABLE}" "-c"
                            "import numpy; print(numpy.__version__)"
                    RESULT_VARIABLE _NUMPY_VERSION_FAIL
                    OUTPUT_VARIABLE NUMPY_VERSION
                    ERROR_VARIABLE _NUMPY_VERSION_ERROR_VALUE
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    IF(NOT _NUMPY_VERSION_FAIL)
        string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" _VER_CHECK "${NUMPY_VERSION}")
        if("${_VER_CHECK}" STREQUAL "")
            # The output from Python was unexpected. Raise an error always
            # here, because we found NumPy, but it appears to be corrupted somehow.
            message(FATAL_ERROR "Requested version from NumPy, got instead:\n${NUMPY_VERSION}\n")
        endif()
            # Get the major and minor version numbers
            string(REGEX REPLACE "\\." ";" _NUMPY_VERSION_LIST ${NUMPY_VERSION})
            list(GET _NUMPY_VERSION_LIST 0 NUMPY_VERSION_MAJOR)
            list(GET _NUMPY_VERSION_LIST 1 NUMPY_VERSION_MINOR)
            list(GET _NUMPY_VERSION_LIST 2 NUMPY_VERSION_PATCH)
    ELSE()
        message(FATAL_ERROR "Could not determine numpy version: ${_NUMPY_VERSION_ERROR_VALUE}")
    ENDIF()
ENDIF()

######################################################################
#
#      status output
#
######################################################################
IF(Python_LIBRARIES AND Python_INCLUDE_DIRS)
    MESSAGE(STATUS "Found Python libraries: ${Python_LIBRARIES}")
    MESSAGE(STATUS "Found Python includes:  ${Python_INCLUDE_DIRS}")
    SET(PYTHONLIBS_FOUND TRUE)
ELSE()
    MESSAGE(FATAL_ERROR "Could NOT find Python libraries and/or includes")
ENDIF()
    
IF(PYTHON_NUMPY_INCLUDE_DIR)
    MESSAGE(STATUS "Python numpy includes: ${PYTHON_NUMPY_INCLUDE_DIR}")
    MESSAGE(STATUS "Python numpy verion: ${NUMPY_VERSION}")
ELSE()
    MESSAGE(FATAL_ERROR "Could NOT find Python numpy ('import numpy.distutils.misc_util' failed)")
ENDIF()

set(LIBDVID_MIN_NUMPY_VERSION "1.7.0")
if (NUMPY_VERSION VERSION_LESS LIBDVID_MIN_NUMPY_VERSION)
    message(FATAL_ERROR "Numpy version (${NUMPY_VERSION}) is too old.  libdvid requires numpy >= ${LIBDVID_MIN_NUMPY_VERSION}")
endif()

######################################################################
#
#      configure package
#
######################################################################

include_directories(${Python_INCLUDE_DIRS} ${PYTHON_NUMPY_INCLUDE_DIR})

add_library( _dvid_python SHARED src/libdvid_python.cpp )

target_link_libraries( _dvid_python dvidcpp ${support_LIBS} ${Boost_PYTHON_LIBRARY} ${PYTHON_LIBRARIES} )
set_target_properties( _dvid_python PROPERTIES PREFIX  ""
                                               DEBUG_POSTFIX "" )

if (APPLE)
    set_target_properties( _dvid_python PROPERTIES SUFFIX  ".so")
endif()

if (NOT (APPLE OR MSVC))
    # FIXME: This may not be needed anymore because timing now uses std::chrono
    target_link_libraries( _dvid_python rt)
endif()

install(TARGETS _dvid_python
        RUNTIME DESTINATION ${LIBDVID_PYTHON_INSTALL_DIR}/libdvid
        LIBRARY DESTINATION ${LIBDVID_PYTHON_INSTALL_DIR}/libdvid)

install(FILES libdvid/__init__.py
              libdvid/mask_codec.py
        DESTINATION ${LIBDVID_PYTHON_INSTALL_DIR}/libdvid)

install(FILES libdvid/util/__init__.py
              libdvid/util/roi_utils.py
              libdvid/util/view_as_blocks.py
        DESTINATION ${LIBDVID_PYTHON_INSTALL_DIR}/libdvid/util)

install(FILES libdvid/gui/__init__.py
              libdvid/gui/contents_browser.py
        DESTINATION ${LIBDVID_PYTHON_INSTALL_DIR}/libdvid/gui)

install(FILES libdvid/voxels/__init__.py
              libdvid/voxels/voxels_accessor.py
              libdvid/voxels/voxels_metadata.py
        DESTINATION ${LIBDVID_PYTHON_INSTALL_DIR}/libdvid/voxels)

######################################################################
#
#      specify python tests
#
######################################################################
add_test(
    python_test_connection
    ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_connection.py
)

add_test(
    python_test_node_service
    ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_node_service.py
)
        
add_test(
    test_roi_utils
    ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_roi_utils.py
)

add_test(
    test_label_codec
    ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_label_codec.py
)

add_test(
    test_mask_codec
    ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mask_codec.py
)

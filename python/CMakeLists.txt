cmake_minimum_required (VERSION 2.8) 

if (NOT ${BUILDEM_DIR} STREQUAL "None")

    # Initialize these variables with paths from BuildEM
    set(PYTHON_INCLUDE_PATH ${PYTHON_INCLUDE_PATH} CACHE PATH "Path to Python include files")
    set(PYTHON_EXECUTABLE ${PYTHON_EXE} CACHE PATH "Path to python executable")
    set(PYTHON_LIBRARIES ${PYTHON_LIBRARY_FILE} CACHE PATH "path to libpython so/dylib")
    set(PYTHON_NUMPY_INCLUDE_DIR "${PYTHON_PREFIX}/lib/python2.7/site-packages/numpy/core/include" CACHE PATH "path to numpy/core/include")
    set(LIBDVID_PYTHON_INSTALL_DIR "${PYTHON_PREFIX}/lib/python2.7/site-packages" CACHE PATH "where to install the libdvid Python package")

else() # Not using buildem -- detect python package details.

    ######################################################################
    #
    #      find Python interpreter
    #
    ######################################################################
    
    FIND_PACKAGE(PythonInterp 2 REQUIRED)
    
    IF(PYTHONINTERP_FOUND)
        # check that Python version 2.x is used
        execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
                             "import sys; print(sys.version[0])"
                              OUTPUT_VARIABLE PYTHON_MAJOR_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
        IF(NOT (${PYTHON_MAJOR_VERSION} EQUAL 2))
            MESSAGE(FATAL_ERROR "Python bindings require Python 2.x.")
        ENDIF()
    ENDIF()
    
    ######################################################################
    #
    #      find Python library
    #
    ######################################################################
    
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
                     "import sys; print sys.exec_prefix"
                      OUTPUT_VARIABLE PYTHON_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    
    IF(APPLE AND ${PYTHON_PREFIX} MATCHES ".*framework.*")
        SET(PYTHON_LIBRARIES "${PYTHON_PREFIX}/Python"
            CACHE FILEPATH "Python libraries"
            FORCE)
    ELSE()
        execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
                         "import sys; skip = 2 if sys.platform.startswith('win') else 1; print 'python' + sys.version[0:3:skip]"
                          OUTPUT_VARIABLE PYTHON_LIBRARY_NAME OUTPUT_STRIP_TRAILING_WHITESPACE)
        FIND_LIBRARY(PYTHON_LIBRARIES ${PYTHON_LIBRARY_NAME} HINTS "${PYTHON_PREFIX}" 
                     PATH_SUFFIXES lib lib64 libs DOC "Python libraries")
    ENDIF()
    
    ######################################################################
    #
    #      find Python includes
    #
    ######################################################################
    
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
                    "from distutils.sysconfig import *; print get_python_inc()"
                    OUTPUT_VARIABLE PYTHON_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)
    SET(PYTHON_INCLUDE_PATH ${PYTHON_INCLUDE}
        CACHE PATH "Path to Python include files"
        FORCE)
    
    ######################################################################
    #
    #      find default install directory for Python modules
    #      (usually PYTHONDIR/Lib/site-packages)
    #
    ######################################################################
    IF(NOT DEFINED LIBDVID_PYTHON_INSTALL_DIR OR LIBDVID_PYTHON_INSTALL_DIR MATCHES "^$")
        execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
                         "from distutils.sysconfig import *; print get_python_lib(1)"
                          OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
        FILE(TO_CMAKE_PATH ${PYTHON_SITE_PACKAGES} LIBDVID_PYTHON_INSTALL_DIR)
    ENDIF()
    SET(LIBDVID_PYTHON_INSTALL_DIR ${LIBDVID_PYTHON_INSTALL_DIR}
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
    IF(NOT PYTHON_NUMPY_INCLUDE_DIR)
        # Note: we must suppress possible output of the 'from numpy... import *' command,
        #       because the output cannot be interpreted correctly otherwise
        execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
                         "import sys, os; sys.stdout = open(os.devnull, 'w'); from numpy.distutils.misc_util import *; sys.__stdout__.write(' '.join(get_numpy_include_dirs()))"
                          RESULT_VARIABLE PYTHON_NUMPY_NOT_FOUND
                          OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIR
                          OUTPUT_STRIP_TRAILING_WHITESPACE)
        IF(NOT PYTHON_NUMPY_NOT_FOUND)
            FILE(TO_CMAKE_PATH ${PYTHON_NUMPY_INCLUDE_DIR} PYTHON_NUMPY_INCLUDE_DIR)
        ELSE()
            SET(PYTHON_NUMPY_INCLUDE_DIR "PYTHON_NUMPY_INCLUDE_DIR-NOTFOUND")
        ENDIF()
    ENDIF()
    
    SET(PYTHON_NUMPY_INCLUDE_DIR ${PYTHON_NUMPY_INCLUDE_DIR}
        CACHE PATH "Path to numpy include files" FORCE)
    
    ######################################################################
    #
    #      get numpy version
    #
    ######################################################################
    IF(NOT PYTHON_NUMPY_NOT_FOUND)
        execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
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
    IF(PYTHON_LIBRARIES AND PYTHON_INCLUDE_PATH)
        MESSAGE(STATUS "Found Python libraries: ${PYTHON_LIBRARIES}")
        MESSAGE(STATUS "Found Python includes:  ${PYTHON_INCLUDE_PATH}")
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

endif()

######################################################################
#
#      configure package
#
######################################################################

include_directories(${PYTHON_INCLUDE_PATH} ${PYTHON_NUMPY_INCLUDE_DIR})

add_library( _dvid_python SHARED src/libdvid_python.cpp )

target_link_libraries( _dvid_python dvidcpp ${support_LIBS} ${boostpython_LIB} ${PYTHON_LIBRARIES} )
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
        DESTINATION ${LIBDVID_PYTHON_INSTALL_DIR}/libdvid)

install(FILES libdvid/util/__init__.py
              libdvid/util/roi_utils.py
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
    ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_connection.py
)

add_test(
    python_test_node_service
    ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_node_service.py
)
        

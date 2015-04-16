cmake_minimum_required (VERSION 2.8) 

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
ELSE()
    MESSAGE(FATAL_ERROR "Could NOT find Python numpy ('import numpy.distutils.misc_util' failed)")
ENDIF()

######################################################################
#
#      configure package
#
######################################################################

include_directories(${PYTHON_INCLUDE_PATH} ${PYTHON_NUMPY_INCLUDE_DIR})

if (NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

add_library( _dvid_python SHARED src/libdvid_python.cpp )

target_link_libraries( _dvid_python dvidcpp ${support_LIBS} ${PYTHON_LIBRARIES} )
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
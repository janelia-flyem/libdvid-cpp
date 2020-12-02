cmake_minimum_required (VERSION 3.18)

######################################################################
#
#      find Python interpreter and libraries
#
######################################################################

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

######################################################################
#
#      find default install directory for Python modules
#      (usually PYTHONDIR/Lib/site-packages)
#
######################################################################
SET(LIBDVID_PYTHON_INSTALL_DIR ${Python3_SITEARCH}
    CACHE PATH "where to install the libdvid Python package" FORCE)
# this is the install path relative to CMAKE_INSTALL_PREFIX,
# use this in INSTALL() commands to get packaging right
FILE(RELATIVE_PATH LIBDVID_PYTHON_INSTALL_DIR ${CMAKE_INSTALL_PREFIX} ${LIBDVID_PYTHON_INSTALL_DIR})


######################################################################
#
#      status output
#
######################################################################
IF(Python3_LIBRARIES AND Python3_INCLUDE_DIRS)
    MESSAGE(STATUS "Found Python libraries: ${Python3_LIBRARIES}")
    MESSAGE(STATUS "Found Python includes:  ${Python3_INCLUDE_DIRS}")
    SET(PYTHONLIBS_FOUND TRUE)
ELSE()
    MESSAGE(FATAL_ERROR "Could NOT find Python libraries and/or includes")
ENDIF()

######################################################################
#
#      configure package
#
######################################################################

# Note: For simplicity, Python3_NumPy_INCLUDE_DIR must be provided on the command line!
include_directories(${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIR})

add_library( _dvid_python SHARED src/libdvid_python.cpp )

#
# DO NOT LINK AGAINST libpython
# Instead, assume that bin/python has been statically linked and
# thus all needed symbols will be available during execution.
# Use -undefined dynamic_lookup to avoid linker errors at build time (and defer them to runtime).
# https://github.com/conda-forge/boost-feedstock/issues/70#issuecomment-486398688
# 
target_link_libraries( _dvid_python dvidcpp
                       ${support_LIBS}
                       #${Python3_LIBRARIES} # do not link against libpython
                       ${Boost_PYTHON_LIBRARY}  )

set_target_properties( _dvid_python PROPERTIES PREFIX  ""
                                               DEBUG_POSTFIX "" )

if (APPLE)
    set_property(TARGET _dvid_python APPEND_STRING PROPERTY LINK_FLAGS "-undefined dynamic_lookup") # See above note about libpython
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

# Sadly, there are so many leaks in the Python code
# that ASAN leak-detection produces way too much noise right now.
# Disable leak detection; just use ASAN for bad memory access.
set(ASAN_OPTIONS="detect_leaks=false")

add_test(
    python_test_connection
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_connection.py
)
set_tests_properties(python_test_connection PROPERTIES
    ENVIRONMENT ASAN_OPTIONS="${ASAN_OPTIONS}")

add_test(
    python_test_node_service
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_node_service.py
)
set_tests_properties(python_test_node_service PROPERTIES
    ENVIRONMENT ASAN_OPTIONS="${ASAN_OPTIONS}")

add_test(
    test_roi_utils
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_roi_utils.py
)
set_tests_properties(test_roi_utils PROPERTIES
    ENVIRONMENT ASAN_OPTIONS="${ASAN_OPTIONS}")

add_test(
    test_label_codec
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_label_codec.py
)
set_tests_properties(test_label_codec PROPERTIES
    ENVIRONMENT ASAN_OPTIONS="${ASAN_OPTIONS}")

add_test(
    test_mask_codec
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mask_codec.py
)
set_tests_properties(test_mask_codec PROPERTIES
    ENVIRONMENT ASAN_OPTIONS="${ASAN_OPTIONS}")

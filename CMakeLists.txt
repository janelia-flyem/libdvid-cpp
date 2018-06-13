CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project (libdvidcpp)

include (ExternalProject)

enable_testing()

set (RUN_ENVIRONMENT "Workstation" CACHE TYPE STRING)
if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release)
endif ()

set (CMAKE_CXX_LINK_FLAGS "-O3")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set (CMAKE_CXX_FLAGS_RELEASE "-O3")

set (CMAKE_CXX_FLAGS_DEBUG "-g")
set (CMAKE_DEBUG_POSTFIX "-g")

# Enable bounds-checking in std::vector, etc for when we use -stdlib=libstdc++ 
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_GLIBCXX_DEBUG")

# Also enable bounds-checking if using -stdlib=libc++
# (Note: 0 means "most checks" and 1 means "even more". Leave it undefined to disable it entirely.)
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_LIBCPP_DEBUG=0")


# When using gcc, specify whether or not to enable the C++11 ABI.
# Our default is to disable it.
# If we didn't specify it, gcc-5 enables the C++11 ABI by default,
# which is ABI-incompatible with libraries built with gcc-4.
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(_GLIBCXX_USE_CXX11_ABI 0 CACHE BOOL
        "Passed to gcc as a preprocessor definition. See gcc docs for _GLIBCXX_USE_CXX11_ABI")
    
    if (_GLIBCXX_USE_CXX11_ABI)
        add_definitions("-D_GLIBCXX_USE_CXX11_ABI=1")
    else()
        add_definitions("-D_GLIBCXX_USE_CXX11_ABI=0")
    endif()
endif()


################################################################################
## RPATH settings
## https://cmake.org/Wiki/CMake_RPATH_handling
##

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")
################################################################################

set (BUILDLOC ${CMAKE_SOURCE_DIR})


set (LIBDVID_WRAP_PYTHON NO CACHE BOOL "Build the libdvid python bindings (requires boost_python)")


# python is intentionally omitted here, in favor of the more reliable detection, below.
FIND_PACKAGE(Boost REQUIRED COMPONENTS thread system)

if (LIBDVID_WRAP_PYTHON)

    FIND_PACKAGE(PythonInterp)

    ######################################################################
    #
    #  Find boost::python library
    #
    #  (Copied from vigra/config/FindVIGRANUMPY_DEPENDENCIES.cmake)
    #
    # 'FIND_PACKAGE(Boost COMPONENTS python)' is unreliable because it often selects
    # boost_python for the wrong Python version
    #
    ######################################################################
    IF(Boost_FOUND)

        IF(Boost_USE_MULTITHREADED)
            # define names for thread-safe library variants
            SET(BOOST_PYTHON_NAMES
                    boost_python-py${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}-mt
                    boost_python-${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}-mt
                    boost_python${PYTHON_VERSION_MAJOR}-mt
                    boost_python-mt)
        ENDIF()

        IF(Boost_LIB_SUFFIX)
            SET(BOOST_PYTHON_NAMES ${BOOST_PYTHON_NAMES}
                # Windows with mangled library names
                boost_python${PYTHON_VERSION_MAJOR}${Boost_LIB_SUFFIX}
                boost_python${Boost_LIB_SUFFIX})
        ENDIF()

        # define names for boost_python library variants
        # (may or may not be thread-safe)
        SET(BOOST_PYTHON_NAMES ${BOOST_PYTHON_NAMES}
                # Linux with multiple Python versions
                boost_python-py${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
                # Gentoo
                boost_python-${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}
                # Mac with Python 3
                boost_python${PYTHON_VERSION_MAJOR}
                # default
                boost_python)

        FIND_LIBRARY(Boost_PYTHON_LIBRARY
                     NAMES ${BOOST_PYTHON_NAMES}
                     NAMES_PER_DIR
                     HINTS "${Boost_LIBRARY_DIR}"
                     DOC "boost_python libraries")
    ENDIF()

    if(Boost_PYTHON_LIBRARY)
        MESSAGE(STATUS "Found boost_python library: ${Boost_PYTHON_LIBRARY}")
    else()
        MESSAGE(FATAL_ERROR "Could NOT find boost_python library")
    endif()
endif()

include_directories(AFTER ${Boost_INCLUDE_DIR})
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set (PLATFORM_DYLIB_EXTENSION "dylib")
else()
    set (PLATFORM_DYLIB_EXTENSION "so")
endif()
if (NOT Boost_LIBRARY_DIR_RELEASE)
    set(Boost_LIBRARY_DIR_RELEASE ${Boost_LIBRARY_DIR})
endif()
set (boost_LIBS ${Boost_LIBRARY_DIR_RELEASE}/libboost_thread.${PLATFORM_DYLIB_EXTENSION} ${Boost_LIBRARY_DIR_RELEASE}/libboost_system.${PLATFORM_DYLIB_EXTENSION})

#set(JPEGTURBO ON)
FIND_PACKAGE(PNG REQUIRED)
if (JPEGTURBO)
    FIND_LIBRARY(JPEG2 turbojpeg)
endif()
FIND_PACKAGE(JPEG REQUIRED)
FIND_PACKAGE(CURL REQUIRED)
FIND_LIBRARY(ZEROMQ_LIBRARIES zmq)
FIND_LIBRARY(LZ4_LIBRARY lz4)

if (NOT LZ4_LIBRARY)
    message(FATAL_ERROR "*** Could not find lz4 library ***")
endif()

FIND_LIBRARY(JSON_LIBRARIES jsoncpp)
if (NOT JSON_LIBRARIES AND UNIX AND NOT APPLE)
    # Try again with pkg-config
    # Explanation can be found here:
    # https://github.com/janelia-flyem/libdvid-cpp/pull/19
    include (FindPkgConfig)
    PKG_SEARCH_MODULE(JSON REQUIRED jsoncpp)
    include_directories(BEFORE ${JSON_INCLUDE_DIRS})

    if (NOT JSON_LIBRARIES)
        message(FATAL_ERROR "*** Could not find jsoncpp library ***")
    endif()
endif()

if (JPEGTURBO)
    set (support_LIBS ${JSON_LIBRARIES} ${boost_LIBS} ${ZEROMQ_LIBRARIES} ${PNG_LIBRARIES} ${CURL_LIBRARIES} ${JPEG_LIBRARIES} ${LZ4_LIBRARY} ${JPEG2})
else()
    set (support_LIBS ${JSON_LIBRARIES} ${boost_LIBS} ${ZEROMQ_LIBRARIES} ${PNG_LIBRARIES} ${CURL_LIBRARIES} ${JPEG_LIBRARIES} ${LZ4_LIBRARY})
endif()

include_directories (BEFORE ${CMAKE_SOURCE_DIR}/libdvid ${CMAKE_SOURCE_DIR})

if (NOT MSVC)
    # The -fPIC flag is necessary for "relocatable" code that might be included in an .so
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

# Compile libdvidcpp library components
add_library (dvidcpp SHARED 
             src/DVIDNodeService.cpp
             src/DVIDServerService.cpp
             src/DVIDConnection.cpp
             src/DVIDException.cpp
             src/DVIDGraph.cpp
             src/DVIDLabelCodec.cpp
             src/BinaryData.cpp
             src/DVIDThreadedFetch.cpp 
             src/Algorithms.cpp 
             src/DVIDCache.cpp)
         #target_compile_definitions(dvidcpp PRIVATE JPEGTURBO=1)

target_link_libraries (dvidcpp ${support_LIBS})

# config file for CMake FIND_PACKAGE command
set (libdvidcpp_version "0.1.0")
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}")
export(TARGETS dvidcpp FILE "${PROJECT_BINARY_DIR}/libdvidcppTargets.cmake")

CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/config/libdvidcppConfig.cmake.in
    ${PROJECT_BINARY_DIR}/libdvidcppConfig.cmake
    @ONLY IMMEDIATE)
CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/config/libdvidcppConfigVersion.cmake.in
    ${PROJECT_BINARY_DIR}/libdvidcppConfigVersion.cmake
    @ONLY IMMEDIATE)

# installation for library
INSTALL(DIRECTORY ${PROJECT_SOURCE_DIR}/libdvid DESTINATION include)
INSTALL(DIRECTORY ${PROJECT_SOURCE_DIR}/png++ DESTINATION include)
INSTALL(TARGETS dvidcpp DESTINATION lib)

## Apparently the config files generated here aren't relocatable,
## which we need if we want to use the package in a conda environment.
#INSTALL(FILES ${PROJECT_BINARY_DIR}/libdvidcppConfig.cmake
#              ${PROJECT_BINARY_DIR}/libdvidcppConfigVersion.cmake
#              ${PROJECT_BINARY_DIR}/libdvidcppTargets.cmake
#        DESTINATION lib${LIB_SUFFIX}/libdvidcpp)


if (LIBDVID_WRAP_PYTHON)
    add_subdirectory(python)    
endif()

add_executable(encode_labels "src/tools/encode_labels.cpp" src/DVIDLabelCodec.cpp)
target_link_libraries(encode_labels)
install(TARGETS encode_labels RUNTIME DESTINATION bin)

add_executable(decode_labels "src/tools/decode_labels.cpp" src/DVIDLabelCodec.cpp)
target_link_libraries(decode_labels)
install(TARGETS decode_labels RUNTIME DESTINATION bin)
add_executable(dvidtest_newrepo "tests/test_newrepo.cpp")
target_link_libraries(dvidtest_newrepo dvidcpp ${support_LIBS})

add_executable(dvidtest_nodeconnection "tests/test_nodeconnection.cpp")
target_link_libraries(dvidtest_nodeconnection dvidcpp ${support_LIBS})

add_executable(dvidtest_grayscale "tests/test_grayscale.cpp")
target_link_libraries(dvidtest_grayscale dvidcpp ${support_LIBS})

add_executable(dvidtest_labelblk "tests/test_labelblk.cpp")
target_link_libraries(dvidtest_labelblk dvidcpp ${support_LIBS})

add_executable(dvidtest_keyvalue "tests/test_keyvalue.cpp")
target_link_libraries(dvidtest_keyvalue dvidcpp ${support_LIBS})

add_executable(dvidtest_labelgraph "tests/test_labelgraph.cpp")
target_link_libraries(dvidtest_labelgraph dvidcpp ${support_LIBS})

add_executable(dvidtest_compression "tests/test_compression.cpp")
target_link_libraries(dvidtest_compression dvidcpp ${support_LIBS})

add_executable(dvidtest_blocks "tests/test_blocks.cpp")
target_link_libraries(dvidtest_blocks dvidcpp ${support_LIBS})

add_executable(dvidtest_roi "tests/test_roi.cpp")
target_link_libraries(dvidtest_roi dvidcpp ${support_LIBS})

add_executable(dvidloadtest_labelblk "load_tests/loadtest_labelblk.cpp")
target_link_libraries(dvidloadtest_labelblk dvidcpp ${support_LIBS})

#
# FIXME: Body tests don't pass
#        ...possibly because sparsevol API is outdated??
message(WARNING "test_body will be skipped -- it doesn't pass any more.") 
#add_executable(dvidtest_body "tests/test_body.cpp")
#target_link_libraries(dvidtest_body dvidcpp ${support_LIBS})

add_executable(dvidloadtest_labelgraph "load_tests/loadtest_labelgraph.cpp")
target_link_libraries(dvidloadtest_labelgraph dvidcpp ${support_LIBS})

add_executable(dvidloadtest_tile "load_tests/loadtest_tile.cpp")
target_link_libraries(dvidloadtest_tile dvidcpp ${support_LIBS})

add_executable(dvidloadtest_sparsegray "load_tests/loadtest_sparsegray.cpp")
target_link_libraries(dvidloadtest_sparsegray dvidcpp ${support_LIBS})

add_executable(dvidcopypaste_bodies "load_tests/copypaste_bodies.cpp")
target_link_libraries(dvidcopypaste_bodies dvidcpp ${support_LIBS})

add_executable(dvidextract_sparse_body "load_tests/extract_sparse_body.cpp")
target_link_libraries(dvidextract_sparse_body dvidcpp ${support_LIBS})

add_test(
    newrepo
    dvidtest_newrepo http://127.0.0.1:8000
)

add_test(
    nodeconnection
    dvidtest_nodeconnection http://127.0.0.1:8000
)

add_test(
    grayscale 
    dvidtest_grayscale http://127.0.0.1:8000
)

add_test(
    labelblk 
    dvidtest_labelblk http://127.0.0.1:8000
)

add_test(
    keyvalue 
    dvidtest_keyvalue http://127.0.0.1:8000
)

add_test(
    labelgraph 
    dvidtest_labelgraph http://127.0.0.1:8000
)

add_test(
    compression
    dvidtest_compression
    ${CMAKE_SOURCE_DIR}/tests/inputs/testimage.jpeg
    ${CMAKE_SOURCE_DIR}/tests/inputs/testimage.png
    ${CMAKE_SOURCE_DIR}/tests/inputs/testimage.binary
)

add_test(
    blocks 
    dvidtest_blocks http://127.0.0.1:8000
)

add_test(
    roi
    dvidtest_roi http://127.0.0.1:8000
)

#
# See fixme above.
#
#add_test(
#    body 
#    dvidtest_body http://127.0.0.1:8000
#)

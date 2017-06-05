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

################################################################################
# Check if BUILDEM_DIR has already been assigned.  If not, create a default.
set (BUILDEM_DIR "None" CACHE TYPE STRING)

if (${BUILDEM_DIR} STREQUAL "None")
    message ("WARNING: To use Buildem, Buildem directory (for all downloads & builds) should be specified via -DBUILDEM_DIR=<path> on cmake command line.")
    message ("Builds will be placed here: ${CMAKE_SOURCE_DIR}/bin")
else()
    message ("FlyEM downloads and builds will be placed here: ${BUILDEM_DIR}")
endif ()
###############################################################################

if (${BUILDEM_DIR} STREQUAL "None")
    set (BUILDLOC ${CMAKE_SOURCE_DIR})
else()
    set (BUILDLOC ${BUILDEM_DIR})
endif()


set (LIBDVID_WRAP_PYTHON NO CACHE BOOL "Build the libdvid python bindings (requires boost_python)")

if (NOT ${BUILDEM_DIR} STREQUAL "None")
    ###############################################################################
    # Download and install buildem, if it isn't already in BUILDEM_DIR.
    set (BUILDEM_REPO_DIR ${BUILDEM_DIR}/src/buildem)
    if (NOT EXISTS ${BUILDEM_REPO_DIR}/python.cmake)
        message ("Installing buildem repo...")
        ExternalProject_Add(buildem
            PREFIX ${BUILDEM_DIR}
            GIT_REPOSITORY http://github.com/janelia-flyem/buildem.git
            UPDATE_COMMAND ""
            PATCH_COMMAND ""
            CONFIGURE_COMMAND "" 
            BUILD_COMMAND ""
            BUILD_IN_SOURCE 1
            INSTALL_COMMAND ""
        )
        message ("\n**********************************************************\n")
        message ("\nAfter running make, you must re-run the cmake command once")
        message ("buildem has been downloaded!\n")
        message ("\n***********************************************************\n")
        return()
    endif() 
    ###############################################################################

    # Use modules from the downloaded buildem
    set (CMAKE_MODULE_PATH ${BUILDEM_REPO_DIR})
    message("Using cmake modules from ${BUILDEM_REPO_DIR}")

    include (jsoncpp)
    include (libpng)
    include (libjpeg)
    include (libcurl)
    include (lz4)
    include (boost)
    
    set (LIBDVID_DEPS ${jsoncpp_NAME} ${libpng_NAME} ${libcurl_NAME}
        ${libjpeg_NAME} ${lz4_NAME} ${boost_NAME})
    set (boost_LIBS ${BUILDEM_LIB_DIR}/libboost_thread.${BUILDEM_PLATFORM_DYLIB_EXTENSION} ${BUILDEM_LIB_DIR}/libboost_system.${BUILDEM_PLATFORM_DYLIB_EXTENSION})
    if (LIBDVID_WRAP_PYTHON)
        include (python)
        include (numpy)
        set (LIBDVID_DEPS ${LIBDVID_DEPS} ${python_NAME} ${numpy_NAME})
        set(boostpython_LIB ${BUILDEM_LIB_DIR}/libboost_python.${BUILDEM_PLATFORM_DYLIB_EXTENSION})
    endif()
    set (support_LIBS ${json_LIB} 
                      ${boost_LIBS} 
                      ${BUILDEM_LIB_DIR}/libpng.${BUILDEM_PLATFORM_DYLIB_EXTENSION} 
                      ${BUILDEM_LIB_DIR}/libcurl.${BUILDEM_PLATFORM_DYLIB_EXTENSION} 
                      ${BUILDEM_LIB_DIR}/libjpeg.${BUILDEM_PLATFORM_DYLIB_EXTENSION} 
                      ${BUILDEM_LIB_DIR}/liblz4.${BUILDEM_PLATFORM_DYLIB_EXTENSION} )
else ()
    set(BOOST_COMPONENTS thread system)
    if (LIBDVID_WRAP_PYTHON)
        set(BOOST_COMPONENTS ${BOOST_COMPONENTS} python)
    endif()

    FIND_PACKAGE(Boost REQUIRED COMPONENTS ${BOOST_COMPONENTS})
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
    if (LIBDVID_WRAP_PYTHON)
        set(boostpython_LIB ${Boost_LIBRARY_DIR_RELEASE}/libboost_python.${PLATFORM_DYLIB_EXTENSION})
    endif()

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
endif (NOT ${BUILDEM_DIR} STREQUAL "None")

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
             src/BinaryData.cpp
             src/DVIDThreadedFetch.cpp 
             src/Algorithms.cpp 
             src/DVIDCache.cpp)
         #target_compile_definitions(dvidcpp PRIVATE JPEGTURBO=1)

target_link_libraries (dvidcpp ${support_LIBS})

if (NOT ${BUILDEM_DIR} STREQUAL "None")
    add_dependencies (dvidcpp ${LIBDVID_DEPS})
endif()

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

# installation for library (make install not needed for a buildem install)
if (${BUILDEM_DIR} STREQUAL "None")
    INSTALL(DIRECTORY ${PROJECT_SOURCE_DIR}/libdvid DESTINATION include)
    INSTALL(DIRECTORY ${PROJECT_SOURCE_DIR}/png++ DESTINATION include)
    INSTALL(TARGETS dvidcpp DESTINATION lib)

    ## Apparently the config files generated here aren't relocatable,
    ## which we need if we want to use the package in a conda environment.
    #INSTALL(FILES ${PROJECT_BINARY_DIR}/libdvidcppConfig.cmake
    #              ${PROJECT_BINARY_DIR}/libdvidcppConfigVersion.cmake
    #              ${PROJECT_BINARY_DIR}/libdvidcppTargets.cmake
    #        DESTINATION lib${LIB_SUFFIX}/libdvidcpp)
else()
    add_custom_command (
    TARGET dvidcpp
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/libdvid ${BUILDEM_DIR}/include/libdvid
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/png++ ${BUILDEM_DIR}/include/png++
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/libdvidcppConfigVersion.cmake ${BUILDEM_DIR}/lib/libdvidcpp/libdvidcppConfigVersion.cmake
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/libdvidcppConfig.cmake ${BUILDEM_DIR}/lib/libdvidcpp/libdvidcppConfig.cmake
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/libdvidcppTargets.cmake ${BUILDEM_DIR}/lib/libdvidcpp/libdvidcppTargets.cmake
    )
endif()

if (LIBDVID_WRAP_PYTHON)
    add_subdirectory(python)    
endif()

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

add_executable(dvidtest_body "tests/test_body.cpp")
target_link_libraries(dvidtest_body dvidcpp ${support_LIBS})

add_executable(dvidloadtest_labelgraph "load_tests/loadtest_labelgraph.cpp")
target_link_libraries(dvidloadtest_labelgraph dvidcpp ${support_LIBS})

add_executable(dvidloadtest_tile "load_tests/loadtest_tile.cpp")
target_link_libraries(dvidloadtest_tile dvidcpp ${support_LIBS})

add_executable(dvidloadtest_sparsegray "load_tests/loadtest_sparsegray.cpp")
target_link_libraries(dvidloadtest_sparsegray dvidcpp ${support_LIBS})

add_executable(dvidcopypaste_bodies "load_tests/copypaste_bodies.cpp")
target_link_libraries(dvidcopypaste_bodies dvidcpp ${support_LIBS})

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

add_test(
    body 
    dvidtest_body http://127.0.0.1:8000
)

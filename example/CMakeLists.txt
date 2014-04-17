CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project (libdvid_example)

include (ExternalProject)

set (RUN_ENVIRONMENT "Workstation" CACHE TYPE STRING)
if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release)
endif ()

set (CMAKE_CXX_LINK_FLAGS "-O3")
set (CMAKE_CXX_FLAGS_RELEASE "-O3")
set (CMAKE_CXX_FLAGS_DEBUG "-ggdb")
set (CMAKE_DEBUG_POSTFIX "-g")

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

    include (cppnetlib)
    include (libdvidcpp)
    
    set (LIBLOC ${BUILDEM_DIR}/lib)
else()
    # find packages for libdvidcpp and cppnetlib
    find_package ( cppnetlib 0.11.0 REQUIRED )
    find_package ( libdvidcpp 0.1.0 REQUIRED )
endif()

# create example target
include_directories(${LIBDVIDCPP_INCLUDE_DIRS})
add_executable(libdvid_example libdvid_example.cpp)
target_link_libraries(libdvid_example ${LIBLOC}/libjsoncpp.so ${LIBDVIDCPP_LIBRARIES} ${CPPNETLIB_LIBRARIES} ${LIBLOC}/libboost_system.so ${LIBLOC}/libboost_thread.so ${LIBLOC}/libssl.so ${LIBLOC}/libcrypto.so)

# installation for non buildem
if (${BUILDEM_DIR} STREQUAL "None")
    INSTALL(TARGETS libdvid_example DESTINATION bin)
else()
    add_dependencies (libdvid_example ${libdvidcpp_NAME} ${cppnetlib_NAME})
endif()


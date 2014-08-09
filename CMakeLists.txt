CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project (libdvidcpp)

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

if (${BUILDEM_DIR} STREQUAL "None")
    set (BUILDLOC ${CMAKE_SOURCE_DIR})
else()
    set (BUILDLOC ${BUILDEM_DIR})
endif()

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
    include (cppnetlib)
    
    include_directories (BEFORE ${CPPNETLIB_INCLUDE_DIRS})
    #set (LIBDVID_EXT_LIBS ${CPPNETLIB_LIBRARIES})
    set (LIBDVID_DEPS ${jsoncpp_NAME} ${cppnetlib_NAME})
else ()
    FIND_PACKAGE(Boost 1.54)
    FIND_PACKAGE(cppnetlib 0.11.0 REQUIRED)
    include_directories(${Boost_INCLUDE_DIRS})
    include_directories (BEFORE ${CPPNETLIB_INCLUDE_DIRS})
    #set (LIBDVID_EXT_LIBS ${CPPNETLIB_LIBRARIES})
endif (NOT ${BUILDEM_DIR} STREQUAL "None")

include_directories (BEFORE ${CMAKE_SOURCE_DIR}/libdvid)

# Compile libdvidcpp library components
add_library (dvidcpp src/DVIDNode.cpp src/DVIDServer.cpp src/Utilities.cpp src/DVIDGraph.cpp)
target_link_libraries (dvidcpp ${LIBDVID_EXT_LIBS})
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
    INSTALL(TARGETS dvidcpp DESTINATION lib)
    INSTALL(FILES ${PROJECT_BINARY_DIR}/libdvidcppConfig.cmake
    ${PROJECT_BINARY_DIR}/libdvidcppConfigVersion.cmake
    DESTINATION lib${LIB_SUFFIX}/libdvidcpp)
else()
    add_custom_command (
    TARGET dvidcpp
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/libdvid ${BUILDEM_DIR}/include/libdvid
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/libdvidcppConfigVersion.cmake ${BUILDEM_DIR}/lib/libdvidcpp/libdvidcppConfigVersion.cmake
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/libdvidcppConfig.cmake ${BUILDEM_DIR}/lib/libdvidcpp/libdvidcppConfig.cmake
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/libdvidcppTargets.cmake ${BUILDEM_DIR}/lib/libdvidcpp/libdvidcppTargets.cmake
    )
endif()



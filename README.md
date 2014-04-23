# libdvid-cpp [![Picture](https://raw.github.com/janelia-flyem/janelia-flyem.github.com/master/images/gray_janelia_logo.png)](http://janelia.org/)

libdvid-cpp provides a c++ wrapper to HTTP calls to [DVID](https://github.com/janelia-flyem/dvid).
It exposes only part of the DVID REST API allowing the setting and
retrieving of nd-data and key-value pairs.

## Installation

The primary dependencies are Boost (>=1.54), cpp-netlib, and jsoncpp.  These dependencies are
resolved automatically if building with [buildem](https://github.com/janelia-flyem/buildem).

There are two preferred ways to build libdvid-cpp both use buildem: 1) build the library directly using buildem (see following) or 2) build the library as a buildem dependency when building your application (see next section).

### buildem installation
    
    % mkdir build; cd build;
    % cmake .. -DBUILDEM_DIR=/user-defined/path/to/build/directory
    % make -j num_processors

This will install the libray to ${BUILDEM_DIR}/lib and cmake modules to ${BUILDEM_DIR}/lib/libdvid.

### stand-alone installation

    % mkdir build; cd build;
    % cmake ..
    % make; make install

cpp-netlib must be installed and in the cmake search path.  jsoncpp must also be installed.

## Building an application

There are two recommended ways for linking lidvid-cpp.  Both are demonstrated in *example/CmakeLists.txt*.  Once libdvid-cpp is found or loaded the following should be added to your CMakeLists.txt:

    % include_directories(${LIBDVIDCPP_INCLUDE_DIRS})
    % target_link_libraries(MYAPP ${LIBLOC}/libjsoncpp.so ${LIBDVIDCPP_LIBRARIES} ${CPPNETLIB_LIBRARIES} ${LIBLOC}/libboost_system.so ${LIBLOC}/libboost_thread.so ${LIBLOC}/libssl.so ${LIBLOC}/libcrypto.so

Where MYAPP is the app you are building and LIBLOC is the location(s) of the listed libraries.

### Find with CMake

Add the following lines to your CMakelists.txt file

    % find_package ( cppnetlib 0.11.0 REQUIRED )
    % find_package ( libdvidcpp 0.1.0 REQUIRED )
    
### buildem application

Follow the documentation in [buildem](https://github.com/janelia-flyem/buildem) and the example in this package to load buildem and add the following lines your CMakeLists.txt:
    
    % set (LIBLOC ${BUILDEM_DIR}/lib) 
    % include (cppnetlib)
    % include (libdvidcpp)
    % add_dependencies (MYAPP ${libdvidcpp_NAME} ${cppnetlib_NAME})
    
The advantage of this approach is that you never have to explicitly build libdvid-cpp.  It is automatically built by buildem as an explicit dependency of your application.
    
## Example application

Consult the application in *example* to see many of the API features exercised.  Applications should include \<libdvid/DVIDNode.h\>.  The API requires the creation of a DVID server type and a DVID node type corresponding to the htpp address where DVID is running and the UUID for the version of the desired repository.  Beyond this state, all API calls are stateless.
    
## TODO

* Add test regressions and more documentation
* Improve performance of streaming large amounts of data by eliminating unnecessary copying.
* Expand support to more datatype instances in DVID (like sparse volumes) and handling DVID meta-data
* Simpler package linking

## Notes
    
* Will not work if datatype instance is 2D (all requests for 2D use the DVID ND interface)
* Currently communicates using uncompressed binary streams

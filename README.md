# libdvid-cpp [![Picture](https://raw.github.com/janelia-flyem/janelia-flyem.github.com/master/images/HHMI_Janelia_Color_Alternate_180x40.png)](http://www.janelia.org)

libdvid-cpp provides a c++ wrapper to HTTP calls to
[DVID](https://github.com/janelia-flyem/dvid).
It exposes only part of the DVID REST API.  Extra functionality
will be added as needed.  Some of this functionality has been exposed
through a Python API.  For examples of how to use the C++ and python API, please consult the unit tests.

## Installation

The primary dependencies are:

* Boost
* [jsoncpp](https://github.com/open-source-parsers/jsoncpp.git)
* libpng
* libcurl
* [lz4](https://github.com/Cyan4973/lz4)
* numpy (>=1.7) (when building the python wrapper)

To build bindings for Python, please add the cmake flag -DLIBDVID_WRAP_PYTHON=1.

To run the regression tests, type "make test" after building.  To successfully
run, DVID must be installed on 127.0.0.1:8000.

### standalone installation

To install libdvid:

    % mkdir build; cd build;
    % cmake ..
    % make; make install

This will install the library libdvidcpp.a.

### buildem (no longer supported)

### conda installation

The conda build system allows easy 1-step installation of libdvid libraries.

The [Miniconda](http://conda.pydata.org/miniconda.html) tool first needs to installed:

```
# Install miniconda to the prefix of your choice, e.g. /my/miniconda

# LINUX:
wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
bash Miniconda-latest-Linux-x86_64.sh

# MAC:
wget https://repo.continuum.io/miniconda/Miniconda-latest-MacOSX-x86_64.sh
bash Miniconda-latest-MacOSX-x86_64.sh

# Activate conda
CONDA_ROOT=`conda info --root`
source ${CONDA_ROOT}/bin/activate root
```
Once conda is in your system path, call the following to install libdvid-cpp:

    % conda create -n <NAME> -c flyem libdvid-cpp

Conda allows builder to create multiple environments.  To use the python
library, set your PATH to the location of PREFIX/< NAME >/bin.  The libdvidcpp.a
can be found in PREFIX/< NAME >/lib.

## Building an application

Building an application is easy with libdvid.  The following shows the libraries
that need to be linked:

    % g++ myapp.cpp -ldvidcpp -ljsoncpp -lboost_system -lpng -lcurl -ljpeg -llz4

libdvid works well with cmake.  To find the package, add the following to the cmake file:
    
    % find_package ( libdvidcpp )

## Overview of Library
DVID provides an HTTP REST interface.  There is a concept of DVID server
that hosts several repositories (or datasets).  Each repository, contains
several different version nodes.  This is analogous to GIT except that
each version node contains different datatypes with specific interfaces.

libdvid implements an API for the different types of DVID interface.  Currently,
it supports DVIDServerService and DVIDNodeService to access some REST
calls at the server level and version node level respectively.  For example,
libdvid can call the DVID server through the server service to
create a new repository.

    DVIDServerService server("http://mydvidserver");
    string uuid = server.create_new_repo("newrepo", "This is my new repo");

The following retrieves a 3D segmentation with dimension sizes indicates by a vector
DIMS and spatial offset by a vector given by OFFSET:

    DVIDNodeService node("http://mydvidserver", UUID);
    Labels3D segmentation = node.get_labels3D("segmentation", DIMS, OFFSET);

For some simple examples of using libdvid, please review *tests/*.  For
detailed explanation of available API, please examined DVIDNodeService.h
and DVIDServerService.h.  For information on the Python interface,
consult python/src/libdvid_python.cpp and the tests in python/tests.

## Important Notes

To use this library in a multi-threaded environment,
instantiate a new service variable for each thread.  Also, GET and POST
requests should involve less bytes than INT_MAX.  If bigger requests
are needed, the request should be divided into multiple calls

lz4 compression has been implemented in libdvid but has not been tested
against DVID yet.  libdvid also implements a labelblk block get/put that
is not yet supported in DVID.  Eventually, DVID will version the APIs
and this library can then be checked against that.
 
## Testing the Package
libdvid contains unit tests under *tests/* and load tests under *load_tests/*.
The unit tests can be run by:

    % cd make
    % make test

A DVID server needs to be running on 127.0.0.1:8000.  It is important
that the libdvid installation matches the DVID installation.

## TODO

* Add support for sparse volumes datatypes
* Better handling of DVID metadata and verification of interface versions


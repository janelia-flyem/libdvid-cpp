# libdvidcpp [![Picture](https://raw.github.com/janelia-flyem/janelia-flyem.github.com/master/images/gray_janelia_logo.png)](http://janelia.org/)

*Status: In development, not ready for use.*

libdvidcpp provides a c++ wrapper to HTTP calls to [DVID](https://github.com/janelia-flyem/dvid).
It exposes only part of the DVID REST API allowing the setting and
retrieving of nd-data and key-value pairs.

## Installation

(TBD) The primary dependencies are Boost (1.54>=) and cpp-netlib.  These dependencies are
resolved automatically if building with [buildem](https://github.com/janelia-flyem/buildem).

To build libdvidcpp using buildem
    
    % mkdir build; cd build;
    % cmake .. -DBUILDEM_DIR=/user-defined/path/to/build/directory
    % make -j num_processors


## Notes
    
* Will not work if datatype is 2D
* Communicates using uncompressed binary streams (should zip in the future)

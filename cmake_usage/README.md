To use libdvid-cpp in your own project, copy the enclosed
`FindLIBDVIDCPP.cmake` somewhere accessible to your `CMAKE_MODULE_PATH`,
and then use `find_package(LIBDVIDCPP)` in your project's
`CMakeLists.txt` file.

When you invoke cmake to configure your project, you can override the following variables:

```
cmake .. \
        -DLIBDVIDCPP_INCLUDE_DIR="${PREFIX}/include" \
        -DLIBDVIDCPP_LIBRARY="${PREFIX}/lib/libdvidcpp.${DYLIB_EXT}" \
        ...
```

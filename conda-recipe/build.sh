# Depending on our platform, shared libraries end with either .so or .dylib
if [[ `uname` == 'Darwin' ]]; then
    DYLIB_EXT=dylib
else
    DYLIB_EXT=so
fi

BUILD_DIR=${BUILD_DIR-build}

CONFIGURE_ONLY=0
if [[ $1 != "" ]]; then
    if [[ $1 == "--configure-only" ]]; then
        CONFIGURE_ONLY=1
    else
        echo "Unknown argument: $1"
        exit 1
    fi
fi


# CONFIGURE
mkdir -p "${BUILD_DIR}" # Using -p here is convenient for calling this script outside of conda.
cd "${BUILD_DIR}"
cmake ..\
        -DCMAKE_C_COMPILER="${PREFIX}/bin/gcc" \
        -DCMAKE_CXX_COMPILER="${PREFIX}/bin/g++" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
        -DCMAKE_PREFIX_PATH="${PREFIX}" \
        -DCMAKE_CXX_FLAGS=-I"${PREFIX}/include" \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,${PREFIX}/lib -L${PREFIX}/lib" \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,${PREFIX}/lib -L${PREFIX}/lib" \
        -DBOOST_ROOT="${PREFIX}" \
        -DBoost_LIBRARY_DIR="${PREFIX}/lib" \
        -DBoost_INCLUDE_DIR="${PREFIX}/include" \
        -DCMAKE_MACOSX_RPATH=ON \
        -DPYTHON_EXECUTABLE="${PYTHON}" \
        -DPYTHON_LIBRARY="${PREFIX}/lib/libpython2.7.${DYLIB_EXT}" \
        -DPYTHON_INCLUDE_DIR="${PREFIX}/include/python2.7" \
        -DLIBDVID_WRAP_PYTHON=1 \
##

if [[ $CONFIGURE_ONLY == 0 ]]; then
    # BUILD
    if [[ $(uname) == 'Darwin' ]]; then
        make -j${CPU_COUNT} 2> >(python "${RECIPE_DIR}"/filter-macos-linker-warnings.py)
    else
        make -j${CPU_COUNT}
    fi


    # "install" to the build prefix (conda will relocate these files afterwards)
    make install
    
    # For debug builds, this symlink can be useful...
    #cd ${PREFIX}/lib && ln -s libdvidcpp-g.${DYLIB_EXT} libdvidcpp.${DYLIB_EXT} && cd -

    if [[ -z "$SKIP_LIBDVID_TESTS" || "$SKIP_LIBDVID_TESTS" == "0" ]]; then
        echo "Running build tests.  To skip, set SKIP_LIBDVID_TESTS=1"
    
        # This script runs 'make test', which uses the build artifacts in the build directory, not the installed files.
        # Therefore, they haven't been post-processed by conda to automatically locate their dependencies.
        # We'll set LD_LIBRARY_PATH to avoid errors from ld
        if [[ $(uname) == Darwin ]]; then
            export DYLD_FALLBACK_LIBRARY_PATH="${PREFIX}/lib":"${DYLD_FALLBACK_LIBRARY_PATH}"
        else
            export LD_LIBRARY_PATH="${PREFIX}/lib":"${LD_LIBRARY_PATH}"
        fi

        if ! make test; then
            cat Testing/Temporary/LastTest.log
            1>&2 echo "****************************************"
            1>&2 echo "Post-build tests FAILED.  See log above."
            1>&2 echo "****************************************"
            exit 1
        fi
    fi
fi

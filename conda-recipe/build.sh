set -x

# Depending on our platform, shared libraries end with either .so or .dylib
if [[ $(uname) == 'Darwin' ]]; then
    DYLIB_EXT=dylib
    CXXFLAGS="${CXXFLAGS} -I${PREFIX}/include -stdlib=libc++"
else
    DYLIB_EXT=so
    CXXFLAGS="${CXXFLAGS} -I${PREFIX}/include"

    # Don't specify these -- let conda-build do it.
    #CC=gcc
    #CXX=g++
fi

CONDA_PY=${CONDA_PY-$(python -c "import sys; print('{}{}'.format(*sys.version_info[:2]))")}

PY_VER=$(python -c "import sys; print('{}.{}'.format(*sys.version_info[:2]))")
PY_ABIFLAGS=$(python -c "import sys; print('' if sys.version_info.major == 2 else sys.abiflags)")
PY_ABI=${PY_VER}${PY_ABIFLAGS}

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

NUMPY_INCLUDE=$(${PREFIX}/bin/python3 -c 'import numpy; print(numpy.get_include())')

# On Mac, you can specify CMAKE_GENERATOR=Xcode if you want.
CMAKE_GENERATOR=${CMAKE_GENERATOR-Unix Makefiles}
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE-Release}

export MACOSX_DEPLOYMENT_TARGET=10.9

#
# For some reason cmake's FindPython3 module seems
# to not work properly at all, at least on macOS-10.15 with cmake-3.18.
# WTF? OK, I'm just overriding everything.
#
PYTHON_CMAKE_SETTINGS=(
    -DPython3_ROOT_DIR=${PREFIX}
    -DPython3_FIND_FRAMEWORK=NEVER
    -DPython3_FIND_VIRTUALENV=ONLY
    -DPython3_EXECUTABLE=${PREFIX}/bin/python3
    -DPython3_INCLUDE_DIR=${PREFIX}/include/python${PY_ABI}
    -DPython3_NumPy_INCLUDE_DIR=${NUMPY_INCLUDE}
    -DPython3_LIBRARY=${PREFIX}/lib/libpython${PY_ABI}.${DYLIB_EXT}
    -DPython3_LIBRARY_RELEASE=${PREFIX}/lib/libpython${PY_ABI}.${DYLIB_EXT}
)

# CONFIGURE
mkdir -p "${BUILD_DIR}" # Using -p here is convenient for calling this script outside of conda.
cd "${BUILD_DIR}"
cmake ..\
    -G "${CMAKE_GENERATOR}" \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_PREFIX_PATH="${PREFIX}" \
    -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
    -DCMAKE_OSX_SYSROOT="${CONDA_BUILD_SYSROOT}" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET}" \
    -DCMAKE_OSX_ARCHITECTURES="$(uname -m)" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L${PREFIX}/lib ${LDFLAGS}" \
    -DCMAKE_EXE_LINKER_FLAGS="-L${PREFIX}/lib ${LDFLAGS}" \
    -DBOOST_ROOT="${PREFIX}" \
    -DBoost_LIBRARY_DIR="${PREFIX}/lib" \
    -DBoost_INCLUDE_DIR="${PREFIX}/include" \
    -DBoost_PYTHON_LIBRARY="${PREFIX}/lib/libboost_python${CONDA_PY}.${DYLIB_EXT}" \
    -DZLIB_INCLUDE_DIR=${PREFIX}/include \
    -DZLIB_LIBRARY=${PREFIX}/lib/libz${SHLIB_EXT} \
    -DPNG_PNG_INCLUDE_DIR=${PREFIX}/include \
    -DPNG_LIBRARY=${PREFIX}/lib/libpng${SHLIB_EXT} \
    ${PYTHON_CMAKE_SETTINGS[@]} \
    -DLIBDVID_WRAP_PYTHON=1 \
    -DWITH_JPEGTURBO=1 \
    -DWITH_LIBDEFLATE=1 \
##


if [[ $CONFIGURE_ONLY == 0 ]]; then
    ##
    ## BUILD
    ##
    make -j${CPU_COUNT}
    
    ##
    ## INSTALL
    ##
    # "install" to the build prefix (conda will relocate these files afterwards)
    make install
    
    # For debug builds, this symlink can be useful...
    #cd ${PREFIX}/lib && ln -s libdvidcpp-g.${DYLIB_EXT} libdvidcpp.${DYLIB_EXT} && cd -

    ##
    ## TEST
    ##
    if [[ -z "$SKIP_LIBDVID_TESTS" || "$SKIP_LIBDVID_TESTS" == "0" ]]; then
        echo "Running build tests.  To skip, set SKIP_LIBDVID_TESTS=1"

	    # Launch dvid
	    echo "Starting test DVID server..."
	    dvid -verbose serve ${RECIPE_DIR}/dvid-testserver-config.toml &
	    DVID_PID=$!
	
	    sleep 5;
	    if [ ! pgrep -x > /dev/null ]; then
	        2>&1 echo "*****************************************************"
	        2>&1 echo "Unable to start test DVID server!                    "
	        2>&1 echo "Do you already have a server runnining on port :8000?"
	        2>&1 echo "*****************************************************"
	        exit 2
	    fi 
	    
	    # Kill the DVID server when this script exits
	    trap 'kill -TERM $DVID_PID' EXIT
	    
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

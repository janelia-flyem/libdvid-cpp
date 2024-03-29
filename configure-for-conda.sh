if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <conda-environment-dir> [build-dir]"
    echo ""
    echo "Prerequisite: Create a suitable conda environment for development:"
    echo ""
    echo "    conda create -n libdvid-devel --only-deps libdvid-cpp"
    echo "    conda install -n libdvid-devel libjpeg-turbo clangxx_osx-64 cmake"
    exit 1;
fi

export PREFIX="${1%/}"
export PYTHON="${PREFIX}/bin/python"
export CPU_COUNT=`python -c "import multiprocessing; print(multiprocessing.cpu_count())"`
export PATH="${PREFIX}/bin":$PATH
export NPY_VER=$(python -c 'import numpy; print(numpy.version.version[:3])')
export RECIPE_DIR=$(pwd)/conda-recipe

# Start in the same directory as this script.
cd `dirname $0`

BUILD_DIR=${2-build}

# If the build dir already exists and CMAKE_INSTALL_PREFIX doesn't 
# match the new destination, we need to start from scratch.
if [[ -e "${BUILD_DIR}/CMakeCache.txt" ]]; then

    grep "CMAKE_INSTALL_PREFIX:PATH=$PREFIX" build/CMakeCache.txt > /dev/null 2> /dev/null
    GREP_RESULT=$?
    if [[ $GREP_RESULT == 1 ]]; then
        echo "*** Removing old build directory: ${BUILD_DIR}" 2>&1
        rm -r "${BUILD_DIR}"
    fi
fi

BUILD_DIR="${BUILD_DIR}" NPY_VER=$NPY_VER bash -x -e - ./conda-recipe/build.sh --configure-only
#BUILD_DIR="${BUILD_DIR}" NPY_VER=$NPY_VER bash -x -e - ./conda-recipe/build.sh

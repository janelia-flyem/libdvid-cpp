#!/bin/bash

##
## Build the docs and push them to the gh-pages branch so github displays them.
## Notes:
##   - The new gh-pages branch is pushed to 'origin'
##   - The old gh-pages branch is DELETED.
##   - Files .nojeykll and circle.yml are automatically added to the gh-pages branch.
##   - The work is done in a temporary clone of this repo (in /tmp), to avoid
##     accidentally messing up your working directory.

set -e

repo_dir=$(git rev-parse --show-toplevel)

echo "Building libdvid, assuming your build directory is named 'build'..."

# Make sure a symlink exists
cd ${repo_dir}/python/libdvid
ln -s ../../build/python/_dvid_python.so || true

# Build libdvid
cd ${repo_dir}/build
make

CMAKE_INSTALL_PREFIX_CACHE_ENTRY=$(grep CMAKE_INSTALL_PREFIX CMakeCache.txt)
CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX_CACHE_ENTRY##*PATH=}
LIB_DIR=${CMAKE_INSTALL_PREFIX}/lib

if [[ $(uname) == "Darwin" ]]; then
    export DYLD_FALLBACK_LIBRARY_PATH=${LIB_DIR}
else
    export LD_LIBRARY_PATH=${LIB_DIR}
fi

# Build the docs in the user's repo
echo "Building the docs..."
cd "${repo_dir}"/python/docs
PYTHONPATH=${repo_dir}/python make html

# Record the current commit, so we can log it in the gh-pages commit message
curr_commit=$(git rev-parse HEAD)

# Read the url of the user's 'origin' remote
origin_details=$(git remote -v | grep "^origin\s")
origin_url=$(echo ${origin_details} | python -c "import sys; print(sys.stdin.read().split(' ')[1])") 

# Clone a copy into /tmp/
rm -rf /tmp/libdvid-gh-pages
git clone --depth=1 "file://${repo_dir}" /tmp/libdvid-gh-pages

# Completely erase the old gh-pages branch and start it from scratch
cd /tmp/libdvid-gh-pages
git branch -D gh-pages || true
git checkout --orphan gh-pages
git reset --hard

# Copy the doc output
cp -r "${repo_dir}"/python/docs/build/html/* .

# Github's 'jekyll' builder can't handle directories with underscores,
# but we don't need jekyll anyway. Disable it.
touch .nojekyll

# Copy circle.yml so that circle-ci knows not to build this branch.
#cp -r ${repo_dir}/circle.yml .

# Commit everything to gh-pages
git add .
git commit -m "Docs built from ${curr_commit}"

# Push to github
git remote add remote-origin ${origin_url}
git push -f remote-origin gh-pages

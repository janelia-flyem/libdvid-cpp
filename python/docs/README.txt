This directory contains the Sphinx sources for the libdvid-cpp Python bindings.

REQUIREMENTS:

In addition to libdvid-cpp's dependencies, you'll also
need some extra stuff to build the docs.  These are
available through conda:

- sphinx
- sphinx_rtd_theme

BUILDING:

$ cd python/docs
$ make html

Now in your browser, open build/html/index.html

PUBLISHING:

We publish these docs via github pages. A script automates the process:

$ cd python/docs
$ ./push-to-origin-gh-pages.sh

Caveats:

 - Your cmake build directory for libdvid-cpp must be 'libdvid-cpp/build'
 - This will push to the remote named 'origin'

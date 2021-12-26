import sys
import site
import subprocess


def get_symbols(lib_path):
    cmd = f"nm -gU {lib_path}"
    r = subprocess.run(cmd, capture_output=True, check=True, shell=True)
    return [line.split()[2] for line in r.stdout.splitlines()]


libjpeg_symbols = get_symbols(f"{sys.prefix}/lib/libjpeg.dylib")
libdvid_symbols = get_symbols(f"{sys.prefix}/lib/libdvidcpp.dylib")

site_pkg = site.getsitepackages()
assert len(site_pkg) == 1
libdvid_python_symbols = get_symbols(f"{site_pkg[0]}/libdvid/_dvid_python.so")


# This is important because we link against libjpegturbo, which
# is incompatible with newer versions of libjpeg.  If any of its
# symbols are exposed, they'll conflict with symbols in libjpeg,
# causing undefined results.
assert not (set(libdvid_symbols) & set(libjpeg_symbols)), \
    "libdvid and libjpeg should have no symbols in common!"
assert not (set(libdvid_python_symbols) & set(libjpeg_symbols)), \
    "_dvid_python.so and libjpeg should have no symbols in common!"

import sys
import site
import platform
import subprocess


if platform.system() == "Darwin":
    def get_symbols(lib_path):
        cmd = f"nm -gU {lib_path}"
        r = subprocess.run(cmd, capture_output=True, check=True, shell=True)
        return [line.split()[2] for line in r.stdout.splitlines()]

    libjpeg_symbols = get_symbols(f"{sys.prefix}/lib/libjpeg.dylib")
    libdvid_symbols = get_symbols(f"{sys.prefix}/lib/libdvidcpp.dylib")

else:
    def get_symbols(lib_path):
        cmd = f"nm -gDC {lib_path}"
        r = subprocess.run(cmd, capture_output=True, check=True, shell=True)
        symbols = []
        for line in r.stdout.splitlines():
            if len(line.split()) < 3:
                continue
            value, type, name = line.split()[:3]
            if type == "T":
                symbols.append(name)
        return symbols

    libjpeg_symbols = get_symbols(f"{sys.prefix}/lib/libjpeg.so")
    libdvid_symbols = get_symbols(f"{sys.prefix}/lib/libdvidcpp.so")


site_pkg = site.getsitepackages()
assert len(site_pkg) == 1
libdvid_python_symbols = get_symbols(f"{site_pkg[0]}/libdvid/_dvid_python.so")


# This is important because we link against libjpegturbo, which
# is incompatible with newer versions of libjpeg.  If any of its
# symbols are exposed, they'll conflict with symbols in libjpeg,
# causing undefined results.
common = set(libdvid_symbols) & set(libjpeg_symbols)
assert not common, \
    f"libdvid and libjpeg should have no symbols in common!\n{common}"

common = set(libdvid_python_symbols) & set(libjpeg_symbols)
assert not common, \
    f"_dvid_python.so and libjpeg should have no symbols in common!\n{common}"

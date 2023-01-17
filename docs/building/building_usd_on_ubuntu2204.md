# Building Pixar's USD 22.11 for Ubuntu 22.04

_Last Updated: 2022/01/12_

Building Pixar's USD 22.11 on Ubuntu 22.04 can be difficult. This guide aims to help those who wish to download and compile USD on their system. For most people, [using the Nvidia binaries should suffice and is the recommended option](https://developer.nvidia.com/usd). If those do not work for you, or you wish to have a self-compiled version, this guide is for you.

## Prerequisites

You need:

- Python 3.7 from the Deadsnakes PPA: https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa
- GCC 11
- Cmake
- USD downloaded from the GitHub repository: https://github.com/PixarAnimationStudios/USD

## Python Setup

As of writing, USD targets Python 3.7. On Ubuntu you need to use the [Deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) to get this. You need the following packages:

- python3.7
- python3.7-dev
- libpython3.7
- libpython3.7-dev

Once you have Python 3.7, you need to install `PyOpenGL` and `PySide2`. **You cannot use your normal system `pip` command for this!** The correct command is:

```shell
python3.7 -m pip install PyOpenGL PySide2
```

## Fixing Boost

USD currently targets Boost 1.70 on Linux, which has issues compiling on Ubuntu 22.04. USD supports up to Boost 1.76 on account of issues in MacOS. We can use this to our advantage. Apply the below patchfile to the repository to fix this. 

```
diff --git a/build_scripts/build_usd.py b/build_scripts/build_usd.py
index 5d3861d0a..96dd1c0a4 100644
--- a/build_scripts/build_usd.py
+++ b/build_scripts/build_usd.py
@@ -695,7 +695,7 @@ if MacOS():
     BOOST_URL = "https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.gz"
     BOOST_VERSION_FILE = "include/boost/version.hpp"
 elif Linux():
-    BOOST_URL = "https://boostorg.jfrog.io/artifactory/main/release/1.70.0/source/boost_1_70_0.tar.gz"
+    BOOST_URL = "https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.gz"
     BOOST_VERSION_FILE = "include/boost/version.hpp"
 elif Windows():
     # The default installation of boost on Windows puts headers in a versioned
```

## Building USD

**NOTE: At this time, only a limited number of install options have been tested. YMMV.**

We can use the USD build scripts to build USD as we normally would, but we need to provide some additional options for Python for this to work correctly. If you just need to quickly get this built, use the following command from the USD repository's directory. It builds the USD and the tools including `usdview`, placing them in `~/.local/USD`. If you want to learn more, read on below.

```shell
python3.7m build_scripts/build_usd.py ~/.local/USD \ 
  --tools \
  --usd-imaging \
  --usdview \
  --build-python-info /usr/bin/python3.7m /usr/include/python3.7m /usr/lib/python3.7/config-3.7m-x86_64-linux-gnu/libpython3.7m.so 3.7
```

The important line here is the `--build-python-info` line. This takes, in order, the Python executable, include directory, library, and version. Using the Deadsnakes PPA, these are:

- `PYTHON_EXECUTABLE` : `/usr/bin/python3.7m`
- `PYTHON_INCLUDE_DIR` : `/usr/include/python3.7m`
- `PYTHON_LIBRARY` : `/usr/lib/python3.7/config-3.7m-x86_64-linux-gnu/libpython3.7m.so`
- `PYTHON_VERSION` : `3.7`

Do note that we are using the `pymalloc` versions of Python. The Deadsnakes PPA version of Python 3.7 is compiled using `pymalloc` and `/usr/bin/python3.7` simply symlinks to `/usr/bin/python3.7m`. You could use the symlinks, but there is **NOT** a symlink for `libpython3.7m.so`, so you need to at least provide the direct path to that.

## Afterword

There are a lot of other options for building USD. If you use the command `python3.7m build_scripts/build_usd.py --help` you can get a list of all these commands. Your mileage may vary with compiling these other features. 

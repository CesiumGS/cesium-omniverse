#/bin/bash

# ********************************************************************
# Note: This only works in kit >=105 due to something with python versioning
# ********************************************************************

# Get the nvidia python executable path, it has a few modules needed by the source
# code we're stubbing, so those imports need to be valid

PROJECT_ROOT=`dirname -- "$( readlink -f -- "$0"; )"`
NVIDIA_USD_BINS="$PROJECT_ROOT/extern/nvidia/_build/target-deps/usd/release/bin"
NVIDIA_PYTHON_BINS="$PROJECT_ROOT/extern/nvidia/_build/target-deps/python/bin"
NVIDIA_PYTHON_EXECUTABLE="$NVIDIA_PYTHON_BINS/python3"
NVIDIA_PYTHON_LIBS="$PROJECT_ROOT/extern/nvidia/_build/target-deps/python/lib"
NVIDIA_USD_PYTHON_LIBS="$PROJECT_ROOT/extern/nvidia/_build/target-deps/usd/release/lib/python"
NVIDIA_USDRT_LIBS="$PROJECT_ROOT/extern/nvidia/_build/target-deps/usdrt/_build/linux-x86_64/release/usdrt_only"

FLAT_LIBRARIES_DIR="/tmp/CesiumOmniverseFlatLibs"
CESIUM_OMNI_STUB_PATH="$PROJECT_ROOT/exts/cesium.omniverse/cesium/omniverse/bindings/CesiumOmniversePythonBindings.pyi"
CESIUM_USD_STUB_PATH="$PROJECT_ROOT/exts/cesium.usd.plugins/cesium/usd/plugins/CesiumUsdSchemas/__init__.pyi"

export PYTHONPATH="$NVIDIA_USD_PYTHON_LIBS:$PYTHONPATH"
export PATH="$NVIDIA_USD_BINS:$NVIDIA_PYTHON_LIBS:$NVIDIA_PYTHON_BINS:$NVIDIA_USD_PYTHON_LIBS:$NVIDIA_USDRT_LIBS:$PATH"

echo "Ensuring mypy is installed"
$NVIDIA_PYTHON_EXECUTABLE -m pip install mypy

echo "Building lib files flat in temp dir"
cmake -B build
cmake --build build
cmake --install build --component library --prefix $FLAT_LIBRARIES_DIR

# To find the imports mypy has to be run from the same dir as the object files
cd "$FLAT_LIBRARIES_DIR/lib"

# We call mypy in this strange way to ensure the correct nvidia python executable is used
echo "Generating stubs"
$NVIDIA_PYTHON_EXECUTABLE -c 'from mypy import stubgen; stubgen.main()' -m CesiumOmniversePythonBindings -v
$NVIDIA_PYTHON_EXECUTABLE -c 'from mypy import stubgen; stubgen.main()' -m _cesiumUsdSchemas -v

echo "Copying stubs"
cp "out/CesiumOmniversePythonBindings.pyi" $CESIUM_OMNI_STUB_PATH
cp "out/_cesiumUsdSchemas.pyi" $CESIUM_USD_STUB_PATH

echo "Formatting stubs"
black $CESIUM_OMNI_STUB_PATH
black $CESIUM_USD_STUB_PATH

echo "Cleaning up temp dir"
rm -rf $FLAT_LIBRARIES_DIR

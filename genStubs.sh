#/bin/bash

# ********************************************************************
# Note: This only works in kit >=105 due to something with python versioning
# ********************************************************************

# Get the nvidia python executable path, it has a few modules needed by the source
# code we're stubbing, so those imports need to be valid

PROJECT_ROOT=`dirname -- "$( readlink -f -- "$0"; )"`

NVIDIA_USD_ROOT=$PROJECT_ROOT/extern/nvidia/_build/target-deps/usd/release
NVIDIA_USD_PYTHON_LIBS=$NVIDIA_USD_ROOT/lib/python

NVIDIA_PYTHON_ROOT=$PROJECT_ROOT/extern/nvidia/_build/target-deps/python
NVIDIA_PYTHON_EXECUTABLE=$NVIDIA_PYTHON_ROOT/python

FLAT_LIBRARIES_DIR="/tmp/CesiumOmniverseFlatLibs"
CESIUM_OMNI_STUB_PATH="$PROJECT_ROOT/exts/cesium.omniverse/cesium/omniverse/bindings/CesiumOmniversePythonBindings.pyi"
CESIUM_USD_STUB_PATH="$PROJECT_ROOT/exts/cesium.usd.plugins/cesium/usd/plugins/CesiumUsdSchemas/__init__.pyi"
CESIUM_TESTS_STUB_PATH="$PROJECT_ROOT/exts/cesium.omniverse.cpp.tests/cesium/omniverse/cpp/tests/bindings/CesiumOmniverseCppTestsPythonBindings.pyi"

export PYTHONPATH="$NVIDIA_USD_PYTHON_LIBS:$PYTHONPATH"

echo "Ensuring mypy and black are installed"
$NVIDIA_PYTHON_EXECUTABLE -m pip install mypy==1.6.1 black==23.1.0

echo "Building lib files flat in temp dir"
cmake -B build
cmake --build build --parallel 8
cmake --install build --component library --prefix $FLAT_LIBRARIES_DIR

# To find the imports mypy has to be run from the same dir as the object files
cd "$FLAT_LIBRARIES_DIR/lib"

# We call mypy in this strange way to ensure the correct nvidia python executable is used
echo "Generating stubs"
$NVIDIA_PYTHON_EXECUTABLE -c 'from mypy import stubgen; stubgen.main()' -m CesiumOmniversePythonBindings -v
$NVIDIA_PYTHON_EXECUTABLE -c 'from mypy import stubgen; stubgen.main()' -m _CesiumUsdSchemas -v
$NVIDIA_PYTHON_EXECUTABLE -c 'from mypy import stubgen; stubgen.main()' -m CesiumOmniverseCppTestsPythonBindings -v

echo "Copying stubs"
cp "out/CesiumOmniversePythonBindings.pyi" $CESIUM_OMNI_STUB_PATH
cp "out/_CesiumUsdSchemas.pyi" $CESIUM_USD_STUB_PATH
cp "out/CesiumOmniverseCppTestsPythonBindings.pyi" $CESIUM_TESTS_STUB_PATH

echo "Formatting stubs"
$NVIDIA_PYTHON_EXECUTABLE -m black $CESIUM_OMNI_STUB_PATH
$NVIDIA_PYTHON_EXECUTABLE -m black $CESIUM_USD_STUB_PATH
$NVIDIA_PYTHON_EXECUTABLE -m black $CESIUM_TESTS_STUB_PATH

echo "Cleaning up temp dir"
rm -rf $FLAT_LIBRARIES_DIR

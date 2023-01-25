#!/bin/bash

NVIDIA_USD_BINS="extern/nvidia/_build/target-deps/usd/release/bin"
NVIDIA_USD_PYTHON_LIBS="extern/nvidia/_build/target-deps/usd/release/lib/python"
PROJECT_ROOT=`dirname -- "$( readlink -f -- "$0"; )"`

export PYTHONPATH="$PYTHONPATH:$PROJECT_ROOT/$NVIDIA_USD_PYTHON_LIBS"
export PATH="$PATH:$PROJECT_ROOT/$NVIDIA_USD_BINS"

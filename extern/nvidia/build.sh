#!/bin/bash
set -e
SCRIPT_DIR=$(dirname ${BASH_SOURCE})
cd "$SCRIPT_DIR"
exec "tools/packman/python.sh" scripts/install.py $@

#!/bin/bash

set -e

SCRIPT_DIR=$(dirname ${BASH_SOURCE})

source "$SCRIPT_DIR/tools/packman/python.sh" "$SCRIPT_DIR/scripts/link_app.py" $@ || exit $?

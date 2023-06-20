#!/bin/bash
set -e
SCRIPT_DIR=$(dirname ${BASH_SOURCE})
KIT_DIR="$SCRIPT_DIR/../extern/nvidia/_build/target-deps/kit-sdk"
exec "$KIT_DIR/kit" --enable omni.kit.test --dev --/app/enableStdoutOutput=0 --/exts/omni.kit.test/testExts/0='cesium.omniverse' --ext-folder "$KIT_DIR/extscore" --ext-folder "$KIT_DIR/exts" --ext-folder "$KIT_DIR/apps" --ext-folder "$SCRIPT_DIR/../exts" --/exts/omni.kit.test/testExtApp="$SCRIPT_DIR/../apps/cesium.omniverse.dev.kit" --/exts/omni.kit.test/testExtOutputPath="$SCRIPT_DIR/../_testoutput" --portable-root "$KIT_DIR/" --/telemetry/mode=test "$@"


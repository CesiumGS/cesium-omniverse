import os
import packmanapi
import sys

REPO_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
KIT_SDK_FILE = os.path.join(REPO_ROOT, "deps/kit-sdk.packman.xml")
TARGET_DEPS_FILE = os.path.join(REPO_ROOT, "deps/target-deps.packman.xml")


if __name__ == "__main__":
    platform = sys.argv[2]
    packmanapi.pull(KIT_SDK_FILE, platform=platform)
    packmanapi.pull(TARGET_DEPS_FILE, platform=platform)

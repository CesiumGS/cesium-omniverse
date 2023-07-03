import omni.kit.pipapi
import platform
import os
from typing import Optional
import logging


def is_arm():
    machine = platform.machine()
    return machine.startswith("arm") or machine.startswith("aarch")


def is_windows():
    return platform.system() == "Windows"


def is_linux():
    return platform.system() == "Linux"


def get_wheel_file_name():
    wheel_name: Optional[str] = None

    if is_linux():
        if is_arm():
            wheel_name = (
                "lxml-4.9.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.manylinux_2_24_aarch64.whl"
            )
        else:
            wheel_name = "lxml-4.9.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_24_x86_64.whl"
    elif is_windows():
        wheel_name = "lxml-4.9.2-cp310-cp310-win_amd64.whl"

    return wheel_name


def install_lxml(cesium_extension_location: str):
    logger: logging.Logger = logging.getLogger(__name__)

    wheel_file_name = get_wheel_file_name()

    if wheel_file_name is None:
        logger.error("Could not install lxml")
        return

    wheel_path = os.path.join(cesium_extension_location, "vendor", wheel_file_name)

    omni.kit.pipapi.install(
        package=wheel_path,
        module="lxml",
        use_online_index=False,
        ignore_cache=True,
        ignore_import_check=False,
    )

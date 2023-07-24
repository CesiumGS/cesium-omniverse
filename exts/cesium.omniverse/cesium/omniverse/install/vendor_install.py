import logging
from typing import List
from .wheel_installer import WheelInfo, WheelInstaller


def perform_vendor_install():
    logger = logging.getLogger(__name__)

    # Only vendor wheels for the main Cesium Omniverse extension should be placed here.
    #  This action needs to be mirrored for each extension.
    vendor_wheels: List[WheelInfo] = [
        WheelInfo(
            module="lxml",
            windows_whl="lxml-4.9.2-cp310-cp310-win_amd64.whl",
            linux_x64_whl=(
                "lxml-4.9.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_24_x86_64.whl"
            ),
            linux_aarch_whl=(
                "lxml-4.9.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.manylinux_2_24_aarch64.whl"
            ),
        )
    ]

    for w in vendor_wheels:
        installer = WheelInstaller(w)

        if not installer.install():
            logger.error(f"Could not install wheel for {w.module}")

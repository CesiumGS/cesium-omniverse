from dataclasses import dataclass
import logging
from pathlib import Path
import platform
import omni.kit.app as app
import omni.kit.pipapi
from ..utils.utils import str_is_empty_or_none


@dataclass
class WheelInfo:
    """
    Data class containing the module and wheel file names for each platform.
    """

    module: str
    windows_whl: str
    linux_x64_whl: str
    linux_aarch_whl: str


class WheelInstaller:
    """
    Class for installing wheel files bundled with the extension.
    """

    def __init__(self, info: WheelInfo, extension_module="cesium.omniverse"):
        """
        Creates a new instance of a wheel installer for installing a python package.

        :param info: A WheelInfo data class containing the information for wheel installation.
        :param extension_module: The full module for the extension, if a different extension is using this class.

        :raises ValueError: If any arguments are null or empty strings.
        """
        self._logger = logging.getLogger(__name__)

        if (
            str_is_empty_or_none(info.windows_whl)
            or str_is_empty_or_none(info.linux_x64_whl)
            or str_is_empty_or_none(info.linux_aarch_whl)
        ):
            raise ValueError(f"One or more wheels is missing for {info.module}.")

        self._info = info

        manager = app.get_app().get_extension_manager()
        ext_id = manager.get_extension_id_by_module(extension_module)
        self._vendor_directory_path = Path(manager.get_extension_path(ext_id)).joinpath("vendor")

    def install(self) -> bool:
        """
        Installs the correct wheel for the current platform.

        :return: ``True`` if the installation was successful.
        """

        if platform.system() == "Windows":
            return self._perform_install(self._info.windows_whl)
        else:
            machine = platform.machine()
            if machine.startswith("arm") or machine.startswith("aarch"):
                return self._perform_install(self._info.linux_aarch_whl)
            return self._perform_install(self._info.linux_x64_whl)

    def _perform_install(self, wheel_file_name: str) -> bool:
        """
        Performs the actual installation of the wheel file.

        :param wheel_file_name: The file name of the wheel to install.

        :return: ``True`` if the installation was successful.
        """

        path = self._vendor_directory_path.joinpath(wheel_file_name)

        return omni.kit.pipapi.install(
            package=str(path),
            module=self._info.module,
            use_online_index=False,
            ignore_cache=True,
            ignore_import_check=False,
        )

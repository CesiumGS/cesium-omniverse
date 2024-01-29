from functools import partial
import asyncio
from typing import Optional, List
import logging
import omni.ext
import omni.ui as ui
import omni.kit.ui
from .powertools_window import CesiumPowertoolsWindow
from cesium.omniverse.utils import wait_n_frames, dock_window_async
from cesium.omniverse.install import WheelInfo, WheelInstaller
from .context_menu import ContextMenu

class CesiumPowertoolsExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()

        self._logger = logging.getLogger(__name__)

        self._powertools_window: Optional[CesiumPowertoolsWindow] = None

        self._install_py_dependencies()

    def on_startup(self):
        self._logger.info("Starting Cesium Power Tools...")

        self._setup_menus()
        self._show_and_dock_startup_windows()
        ContextMenu.startup()

    def on_shutdown(self):
        self._destroy_powertools_window()
        ContextMenu.shutdown()

    def _setup_menus(self):
        ui.Workspace.set_show_window_fn(
            CesiumPowertoolsWindow.WINDOW_NAME, partial(self._show_powertools_window, None)
        )

        editor_menu = omni.kit.ui.get_editor_menu()

        if editor_menu:
            editor_menu.add_item(
                CesiumPowertoolsWindow.MENU_PATH, self._show_powertools_window, toggle=True, value=True
            )

    def _show_and_dock_startup_windows(self):
        ui.Workspace.show_window(CesiumPowertoolsWindow.WINDOW_NAME)

        asyncio.ensure_future(dock_window_async(self._powertools_window, target="Property"))

    def _destroy_powertools_window(self):
        if self._powertools_window is not None:
            self._powertools_window.destroy()
        self._powertools_window = None

    async def _destroy_window_async(self, path):
        # Wait one frame, this is due to the one frame defer in Window::_moveToMainOSWindow()
        await wait_n_frames(1)

        if path is CesiumPowertoolsWindow.MENU_PATH:
            self._destroy_powertools_window()

    def _visibility_changed_fn(self, path, visible):
        editor_menu = omni.kit.ui.get_editor_menu()
        if editor_menu:
            editor_menu.set_value(path, visible)
        if not visible:
            asyncio.ensure_future(self._destroy_window_async(path))

    def _show_powertools_window(self, _menu, value):
        if value:
            self._powertools_window = CesiumPowertoolsWindow(width=300, height=400)
            self._powertools_window.set_visibility_changed_fn(
                partial(self._visibility_changed_fn, CesiumPowertoolsWindow.MENU_PATH)
            )
        elif self._powertools_window is not None:
            self._powertools_window.visible = False

    def _install_py_dependencies(self):
        vendor_wheels: List[WheelInfo] = [
            WheelInfo(
                module="pyproj",
                windows_whl="pyproj-3.6.0-cp310-cp310-win_amd64.whl",
                linux_x64_whl="pyproj-3.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
                linux_aarch_whl="pyproj-3.6.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
            ),
            WheelInfo(
                module="numpy",
                windows_whl="numpy-1.26.3-cp310-cp310-win_amd64",
                linux_x64_whl="numpy-1.26.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64",
                linux_aarch_whl="numpy-1.26.3-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64",
            ),
            WheelInfo(
                module="shapely",
                windows_whl="shapely-2.0.2-cp310-cp310-win_amd64.whl",
                linux_x64_whl="shapely-2.0.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
                linux_aarch_whl="shapely-2.0.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
            )
        ]

        for w in vendor_wheels:
            installer = WheelInstaller(w, extension_module="cesium.powertools")

            if not installer.install():
                self._logger.error(f"Could not install wheel for {w.module}")

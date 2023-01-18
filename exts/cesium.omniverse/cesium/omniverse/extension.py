from .bindings import *
from .utils import wait_n_frames
from .ui.asset_window import CesiumOmniverseAssetWindow
from .ui.debug_window import CesiumOmniverseDebugWindow
from .ui.main_window import CesiumOmniverseMainWindow
import asyncio
from functools import partial
import logging
import omni.ext
import omni.kit.ui
import omni.ui as ui
import omni.usd
import os
from typing import Optional, Callable

cesium_extension_location = os.path.join(os.path.dirname(__file__), "../../")

# Global public interface object.
_cesium_omniverse_interface: Optional[ICesiumOmniverseInterface] = None


# Public API.
def get_cesium_omniverse_interface() -> ICesiumOmniverseInterface:
    return _cesium_omniverse_interface


class CesiumOmniverseExtension(omni.ext.IExt):

    @staticmethod
    async def _dock_window_async(window: Optional[ui.Window], target: str = "Stage",
                                 position: ui.DockPosition = ui.DockPosition.SAME):
        if window is None:
            return

        # Wait five frame
        await wait_n_frames(5)
        stage_window = ui.Workspace.get_window(target)
        window.dock_in(stage_window, position, 1)
        window.focus()

    @staticmethod
    def _set_menu(path, value):
        # Set the menu to create this window on and off
        editor_menu = omni.kit.ui.get_editor_menu()
        if editor_menu:
            editor_menu.set_value(path, value)

    def __init__(self) -> None:
        super().__init__()

        self._main_window: Optional[CesiumOmniverseMainWindow] = None
        self._asset_window: Optional[CesiumOmniverseAssetWindow] = None
        self._debug_window: Optional[CesiumOmniverseDebugWindow] = None
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._menu = None

    def on_startup(self):
        # The ability to show up the window if the system requires it. We use it in QuickLayout.
        ui.Workspace.set_show_window_fn(CesiumOmniverseMainWindow.WINDOW_NAME, partial(self.show_main_window, None))
        ui.Workspace.set_show_window_fn(CesiumOmniverseAssetWindow.WINDOW_NAME, partial(self.show_assets_window, None))
        ui.Workspace.set_show_window_fn(CesiumOmniverseDebugWindow.WINDOW_NAME, partial(self.show_debug_window, None))

        show_on_startup = True

        self._add_to_menu(CesiumOmniverseMainWindow.MENU_PATH, self.show_main_window, show_on_startup)
        self._add_to_menu(CesiumOmniverseAssetWindow.MENU_PATH, self.show_assets_window, False)
        self._add_to_menu(CesiumOmniverseDebugWindow.MENU_PATH, self.show_debug_window, False)

        self._logger.info("CesiumOmniverse startup")

        # Acquire the Cesium Omniverse interface.
        global _cesium_omniverse_interface
        _cesium_omniverse_interface = acquire_cesium_omniverse_interface()
        _cesium_omniverse_interface.initialize(cesium_extension_location)

        # Show the window. It will call `self.show_window`
        if show_on_startup:
            ui.Workspace.show_window(CesiumOmniverseMainWindow.WINDOW_NAME)

    def on_shutdown(self):
        self._menu = None

        if self._main_window is not None:
            self._main_window.destroy()
            self._main_window = None

        if self._asset_window is not None:
            self._asset_window.destroy()
            self._asset_window = None

        if self._debug_window is not None:
            self._debug_window.destroy()
            self._debug_window = None

        # Deregister the function that shows the window from omni.ui
        ui.Workspace.set_show_window_fn(CesiumOmniverseMainWindow.WINDOW_NAME, None)
        ui.Workspace.set_show_window_fn(CesiumOmniverseAssetWindow.WINDOW_NAME, None)
        ui.Workspace.set_show_window_fn(CesiumOmniverseDebugWindow.WINDOW_NAME, None)

        self._logger.info("CesiumOmniverse shutdown")

        # Release the Cesium Omniverse interface.
        global _cesium_omniverse_interface
        _cesium_omniverse_interface.finalize()
        release_cesium_omniverse_interface(_cesium_omniverse_interface)
        _cesium_omniverse_interface = None

    def _add_to_menu(self, path, callback: Callable[[bool], None], show_on_startup):
        editor_menu = omni.kit.ui.get_editor_menu()

        if editor_menu:
            self._menu = editor_menu.add_item(path, callback, toggle=True, value=show_on_startup)

    async def _destroy_window_async(self, path):
        # Wait one frame, this is due to the one frame defer in Window::_moveToMainOSWindow()
        await wait_n_frames(1)

        if path is CesiumOmniverseMainWindow.MENU_PATH:
            if self._main_window is not None:
                self._main_window.destroy()
            self._main_window = None
        elif path is CesiumOmniverseAssetWindow.MENU_PATH:
            if self._asset_window is not None:
                self._asset_window.destroy()
            self._asset_window = None
        elif path is CesiumOmniverseDebugWindow.MENU_PATH:
            if self._debug_window is not None:
                self._debug_window.destroy()
            self._debug_window = None

    def _visibility_changed_fn(self, path, visible):
        # Called when the user pressed "X"
        self._set_menu(path, visible)
        if not visible:
            # Destroy the window, since we are creating new window in show_window
            asyncio.ensure_future(self._destroy_window_async(path))

    def show_main_window(self, _menu, value):
        if value:
            self._main_window = CesiumOmniverseMainWindow(
                _cesium_omniverse_interface, width=300, height=400
            )
            self._main_window.set_visibility_changed_fn(
                partial(self._visibility_changed_fn, CesiumOmniverseMainWindow.MENU_PATH))
            asyncio.ensure_future(self._dock_window_async(self._main_window))
        elif self._main_window is not None:
            self._main_window.visible = False

    def show_assets_window(self, _menu, value):
        if value:
            self._asset_window = CesiumOmniverseAssetWindow(
                _cesium_omniverse_interface, width=700, height=300
            )
            self._asset_window.set_visibility_changed_fn(
                partial(self._visibility_changed_fn, CesiumOmniverseAssetWindow.MENU_PATH))
            asyncio.ensure_future(self._dock_window_async(self._asset_window, "Content"))
        elif self._asset_window is not None:
            self._asset_window.visible = False

    def show_debug_window(self, _menu, value):
        if value:
            self._debug_window = CesiumOmniverseDebugWindow(
                _cesium_omniverse_interface, CesiumOmniverseDebugWindow.WINDOW_NAME, width=300, height=365
            )
            self._debug_window.set_visibility_changed_fn(
                partial(self._visibility_changed_fn, CesiumOmniverseDebugWindow.MENU_PATH))
            asyncio.ensure_future(self._dock_window_async(self._debug_window))
        elif self._debug_window is not None:
            self._debug_window.visible = False

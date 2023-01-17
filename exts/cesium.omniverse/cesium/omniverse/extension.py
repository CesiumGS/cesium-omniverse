from .bindings.CesiumOmniversePythonBindings import *
from .window import CesiumOmniverseDebugWindow
from .utils import wait_n_frames
import asyncio
from functools import partial
import logging
import omni.ext
import omni.kit.ui
import omni.ui as ui
import omni.usd
import os
from typing import Optional

cesium_extension_location = os.path.join(os.path.dirname(__file__), "../../")

# Global public interface object.
_cesium_omniverse_interface: ICesiumOmniverseInterface = None

# Public API.
def get_cesium_omniverse_interface() -> ICesiumOmniverseInterface:
    return _cesium_omniverse_interface


class CesiumOmniverseExtension(omni.ext.IExt):

    DEBUG_WINDOW_NAME = "Cesium Debugging"
    MENU_PATH = f"Window/{DEBUG_WINDOW_NAME}"

    _window: Optional[CesiumOmniverseDebugWindow] = None
    _logger: logging.Logger

    def __init__(self) -> None:
        super().__init__()

        self._menu = None
        self._logger = logging.getLogger(__name__)

    def on_startup(self):
        # The ability to show up the window if the system requires it. We use it in QuickLayout.
        ui.Workspace.set_show_window_fn(CesiumOmniverseExtension.DEBUG_WINDOW_NAME, partial(self.show_window, None))

        show_on_startup = True

        # Put the new menu
        editor_menu = omni.kit.ui.get_editor_menu()
        if editor_menu:
            self._menu = editor_menu.add_item(
                CesiumOmniverseExtension.MENU_PATH, self.show_window, toggle=True, value=show_on_startup
            )

        self._logger.info("CesiumOmniverse startup")

        # Acquire the Cesium Omniverse interface.
        global _cesium_omniverse_interface
        _cesium_omniverse_interface = acquire_cesium_omniverse_interface()
        _cesium_omniverse_interface.initialize(cesium_extension_location)

        # Show the window. It will call `self.show_window`
        if show_on_startup:
            ui.Workspace.show_window(CesiumOmniverseExtension.DEBUG_WINDOW_NAME)

    def on_shutdown(self):
        self._menu = None
        if self._window is not None:
            self._window.destroy()
            self._window = None

        # Deregister the function that shows the window from omni.ui
        ui.Workspace.set_show_window_fn(CesiumOmniverseExtension.DEBUG_WINDOW_NAME, None)

        self._logger.info("CesiumOmniverse shutdown")

        # Release the Cesium Omniverse interface.
        global _cesium_omniverse_interface
        _cesium_omniverse_interface.finalize()
        release_cesium_omniverse_interface(_cesium_omniverse_interface)
        _cesium_omniverse_interface = None

    def _set_menu(self, value):
        # Set the menu to create this window on and off
        editor_menu = omni.kit.ui.get_editor_menu()
        if editor_menu:
            editor_menu.set_value(CesiumOmniverseExtension.MENU_PATH, value)

    async def _destroy_window_async(self):
        # Wait one frame, this is due to the one frame defer in Window::_moveToMainOSWindow()
        await wait_n_frames(1)
        if self._window is not None:
            self._window.destroy()
            self._window = None

    async def _dock_window_async(self):
        # Wait five frame
        await wait_n_frames(5)
        if self._window is not None:
            stage_window = ui.Workspace.get_window("Stage")
            self._window.dock_in(stage_window, ui.DockPosition.SAME, 1)
            self._window.focus()

    def _visiblity_changed_fn(self, visible):
        # Called when the user pressed "X"
        self._set_menu(visible)
        if not visible:
            # Destroy the window, since we are creating new window in show_window
            asyncio.ensure_future(self._destroy_window_async())

    def show_window(self, menu, value):
        if value:
            self._window = CesiumOmniverseDebugWindow(
                _cesium_omniverse_interface, CesiumOmniverseExtension.DEBUG_WINDOW_NAME, width=300, height=365
            )
            self._window.set_visibility_changed_fn(self._visiblity_changed_fn)
            asyncio.ensure_future(self._dock_window_async())
        elif self._window is not None:
            self._window.visible = False

import logging
from typing import Optional
import omni.ui as ui
from .troubleshooter_window import CesiumTroubleshooterWindow
from ..bindings import ICesiumOmniverseInterface


class CesiumOmniverseDebugWindow(ui.Window):
    WINDOW_NAME = "Cesium Debugging"
    MENU_PATH = f"Window/Cesium/{WINDOW_NAME}"

    _logger: logging.Logger
    _cesium_omniverse_interface: Optional[ICesiumOmniverseInterface] = None

    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, title: str, **kwargs):
        super().__init__(title, **kwargs)

        self._logger = logging.getLogger(__name__)
        self._cesium_omniverse_interface = cesium_omniverse_interface
        self._cesium_message_field: Optional[ui.SimpleStringModel] = None

        # Set the function that is called to build widgets when the window is visible
        self.frame.set_build_fn(self._build_fn)

    def destroy(self):
        # It will destroy all the children
        super().destroy()

    def _build_fn(self):
        """Builds out the UI buttons and their handlers."""

        def remove_all_tilesets():
            """Removes all tilesets from the stage."""

            tileset_paths = self._cesium_omniverse_interface.get_all_tileset_paths()

            for tileset_path in tileset_paths:
                self._cesium_omniverse_interface.remove_tileset(tileset_path)

        def reload_all_tilesets():
            """Reloads all tilesets."""

            tileset_paths = self._cesium_omniverse_interface.get_all_tileset_paths()

            for tileset_path in tileset_paths:
                self._cesium_omniverse_interface.reload_tileset(tileset_path)

        def print_fabric_stage():
            """Prints the contents of the Fabric stage to a text field."""

            fabric_stage = self._cesium_omniverse_interface.print_fabric_stage()
            self._cesium_message_field.set_value(fabric_stage)

        def open_troubleshooting_window():
            CesiumTroubleshooterWindow(self._cesium_omniverse_interface, "Testing", 1, 0, "Testing")

        with ui.VStack():
            ui.Button("Remove all Tilesets", clicked_fn=lambda: remove_all_tilesets())
            ui.Button("Reload all Tilesets", clicked_fn=lambda: reload_all_tilesets())
            ui.Button("Open Troubleshooter", clicked_fn=lambda: open_troubleshooting_window())
            ui.Button("Print Fabric stage", clicked_fn=lambda: print_fabric_stage())
            with ui.VStack():
                self._cesium_message_field = ui.SimpleStringModel("")
                ui.StringField(self._cesium_message_field, multiline=True, read_only=True)

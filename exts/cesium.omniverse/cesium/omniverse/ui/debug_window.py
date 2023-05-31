import logging
from typing import Optional
import omni.ui as ui
from .statistics_widget import CesiumOmniverseStatisticsWidget
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
        self._cesium_message_field: ui.SimpleStringModel = ui.SimpleStringModel("")
        self._statistics_widget: Optional[CesiumOmniverseStatisticsWidget] = None

        # Set the function that is called to build widgets when the window is visible
        self.frame.set_build_fn(self._build_fn)

    def destroy(self):
        if self._statistics_widget is not None:
            self._statistics_widget.destroy()
            self._statistics_widget = None

        # It will destroy all the children
        super().destroy()

    def __del__(self):
        self.destroy()

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

        with ui.VStack(spacing=10):
            with ui.VStack():
                ui.Button("Remove all Tilesets", height=20, clicked_fn=remove_all_tilesets)
                ui.Button("Reload all Tilesets", height=20, clicked_fn=reload_all_tilesets)
                ui.Button("Print Fabric stage", height=20, clicked_fn=print_fabric_stage)
                ui.StringField(self._cesium_message_field, height=100, multiline=True, read_only=True)
            self._statistics_widget = CesiumOmniverseStatisticsWidget(self._cesium_omniverse_interface)

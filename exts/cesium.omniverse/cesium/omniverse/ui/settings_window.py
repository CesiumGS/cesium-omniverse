import logging
import carb.settings
from typing import Optional
import omni.ui as ui
from ..bindings import ICesiumOmniverseInterface
from cesium.omniverse.utils.custom_fields import int_field_with_label


class CesiumOmniverseSettingsWindow(ui.Window):
    WINDOW_NAME = "Cesium Settings"
    MENU_PATH = f"Window/Cesium/{WINDOW_NAME}"

    _logger: logging.Logger
    _cesium_omniverse_interface: Optional[ICesiumOmniverseInterface] = None

    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, title: str, **kwargs):
        super().__init__(title, **kwargs)

        self._logger = logging.getLogger(__name__)
        self._cesium_omniverse_interface = cesium_omniverse_interface
        self._cache_items_setting = "/persistent/exts/cesium.omniverse/maxCacheItems"

        # Set the function that is called to build widgets when the window is visible
        self.frame.set_build_fn(self._build_fn)

    def destroy(self):
        # It will destroy all the children
        super().destroy()

    def __del__(self):
        self.destroy()

    @staticmethod
    def show_window():
        ui.Workspace.show_window(CesiumOmniverseSettingsWindow.WINDOW_NAME)

    def _build_fn(self):
        """Builds out the UI buttons and their handlers."""

        def set_cache_parameters():
            newval = self._cache_items_model.get_value_as_int()
            carb.settings.get_settings().set(self._cache_items_setting, newval)

        def clear_cache():
            self._cesium_omniverse_interface.clear_accessor_cache()

        with ui.VStack(spacing=4):
            cache_items = carb.settings.get_settings().get(self._cache_items_setting)
            self._cache_items_model = ui.SimpleIntModel(cache_items)
            int_field_with_label("Maximum cache items", model=self._cache_items_model)
            ui.Button("Set cache parameters (requires restart)", height=20, clicked_fn=set_cache_parameters)
            ui.Button("Clear cache", height=20, clicked_fn=clear_cache)

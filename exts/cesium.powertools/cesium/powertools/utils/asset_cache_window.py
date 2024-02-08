import logging
import carb.settings
import omni.ui as ui
from cesium.omniverse.utils.custom_fields import int_field_with_label

class CesiumAssetCacheWindow(ui.Window):
    WINDOW_NAME = "Cesium Asset Cache"

    _logger: logging.Logger

    def __init__(self, **kwargs):
        super().__init__(CesiumAssetCacheWindow.WINDOW_NAME, **kwargs)

        self._logger = logging.getLogger(__name__)

        # Set the function that is called to build widgets when the window is visible
        self.frame.set_build_fn(self._build_fn)

        self._cache_items_setting = "/persistent/exts/cesium.omniverse/maxCacheItems"

    def destroy(self):
        # It will destroy all the children
        super().destroy()

    def __del__(self):
        self.destroy()

    @staticmethod
    def create_window():
        return CesiumAssetCacheWindow(width=250, height=400)

    def _set_cache_parameters(self):
        newval = self._cache_items_model.get_value_as_int()
        carb.settings.get_settings().set(self._cache_items_setting, newval)

    def _build_fn(self):
        """Builds out the UI buttons and their handlers."""

        with ui.VStack(spacing=4):
            label_style = {"Label": {"font_size": 16}}
            cache_items = carb.settings.get_settings().get(self._cache_items_setting)
            self._cache_items_model = ui.SimpleIntModel(cache_items)
            int_field_with_label("Maximum cache items", model=self._cache_items_model)
            ui.Button("Set cache parameters", height=20, clicked_fn=self._set_cache_parameters)

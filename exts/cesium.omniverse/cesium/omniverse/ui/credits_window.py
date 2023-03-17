import logging
import omni.kit.app as app
import omni.ui as ui
from pathlib import Path
from typing import List
from .credits_parser import CesiumCreditsParser
from ..bindings import ICesiumOmniverseInterface
from .styles import CesiumOmniverseUiStyles


class CesiumOmniverseCreditsWindow(ui.Window):
    WINDOW_NAME = "Data Attribution"

    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, **kwargs):
        super().__init__(CesiumOmniverseCreditsWindow.WINDOW_NAME, **kwargs)

        manager = app.get_app().get_extension_manager()
        ext_id = manager.get_extension_id_by_module("cesium.omniverse")

        self._cesium_omniverse_interface = cesium_omniverse_interface
        self._logger = logging.getLogger(__name__)
        self._images_path = Path(manager.get_extension_path(ext_id)).joinpath("images")

        self.height = 500
        self.width = 400

        self.padding_x = 12
        self.padding_y = 12

        self._credits: List[(str, bool)] = self._cesium_omniverse_interface.get_credits()

        self.frame.set_build_fn(self._build_ui)

    def __del__(self):
        self.destroy()

    def destroy(self):
        super().destroy()

    def _build_ui(self):
        with ui.VStack(spacing=5):
            ui.Label("Data Provided By:", height=0, style=CesiumOmniverseUiStyles.attribution_header_style)

            CesiumCreditsParser(self._credits, should_show_on_screen=False, perform_fallback=True)

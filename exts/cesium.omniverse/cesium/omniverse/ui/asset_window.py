from ..bindings import ICesiumOmniverseInterface
import logging
import omni.ui as ui


class CesiumOmniverseAssetWindow(ui.Window):
    """
    The asset list window for Cesium for Omniverse. Docked in the same area as "Assets".
    """

    WINDOW_NAME = "Cesium Assets"
    MENU_PATH = f"Window/Cesium/{WINDOW_NAME}"

    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, **kwargs):
        super().__init__(CesiumOmniverseAssetWindow.WINDOW_NAME, **kwargs)

        self._cesium_omniverse_interface = cesium_omniverse_interface
        self._logger = logging.getLogger(__name__)

        self.frame.set_build_fn(self._build_fn)

    def destroy(self):
        super().destroy()

    def _build_fn(self):
        """Builds all UI components."""

        with ui.VStack():
            ui.Label("TODO: The rest of the Cesium assets window.")

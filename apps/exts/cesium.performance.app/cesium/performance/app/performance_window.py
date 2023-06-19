import logging
import omni.ui as ui


class CesiumPerformanceWindow(ui.Window):
    WINDOW_NAME = "Cesium Performance Testing"
    MENU_PATH = f"Window/Cesium/{WINDOW_NAME}"

    def __init__(self, **kwargs):
        super().__init__(CesiumPerformanceWindow.WINDOW_NAME, **kwargs)

        self._logger = logging.getLogger(__name__)

        self.frame.set_build_fn(self._build_fn)

    def destroy(self) -> None:
        super().destroy()

    def _build_fn(self):
        ui.Label("Performance Testing UI")

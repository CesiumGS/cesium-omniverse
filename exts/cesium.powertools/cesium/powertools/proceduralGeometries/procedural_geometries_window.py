import logging
import omni.ui as ui
from cesium.omniverse.extension import _cesium_omniverse_interface as coi


class ProceduralGeometryWindow(ui.Window):
    WINDOW_NAME = "Procedural Geometry"

    _logger: logging.Logger

    def __init__(self, **kwargs):
        super().__init__(ProceduralGeometryWindow.WINDOW_NAME, **kwargs)

        self._logger = logging.getLogger(__name__)

        # Set the function that is called to build widgets when the window is visible
        self.frame.set_build_fn(self._build_fn)

        self._cesium_omniverse_interface = coi

    def destroy(self):
        # It will destroy all the children
        super().destroy()

    def __del__(self):
        self.destroy()

    @staticmethod
    def create_window():
        return ProceduralGeometryWindow(width=250, height=250)

    def _create_prims(self):
        return_val = self._cesium_omniverse_interface.create_procedural_prims()
        self._logger.info(f"return val is {return_val}")

    def _alter_prims(self):
        return_val = self._cesium_omniverse_interface.alter_procedural_prims()
        self._logger.info(f"return val is {return_val}")

    def _animate_prims(self):
        return_val = self._cesium_omniverse_interface.animate_procedural_prims(.5)
        self._logger.info(f"return val is {return_val}")

    def _build_fn(self):
        """Builds out the UI buttons and their handlers."""

        with ui.VStack(spacing=4):
            label_style = {"Label": {"font_size": 16}}

            ui.Label(
                "Run Procedural Geometry Experiment",
                word_wrap=True,
                style=label_style,
            )

            ui.Button("Create prims", height=20, clicked_fn=self._create_prims)

            ui.Button("Alter prims", height=20, clicked_fn=self._alter_prims)

            ui.Button("Animate prims",  height=20, clicked_fn=self._animate_prims)


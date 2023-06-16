import logging
import omni.ui as ui
from cesium.omniverse.ui.models.space_delimited_number_model import SpaceDelimitedNumberModel

RANDOM_COLORS_TEXT = "Random colors"
FORBID_HOLES_TEXT = "Forbid holes"
FRUSTUM_CULLING_TEXT = "Frustum culling"

NEW_YORK_CITY_TEXT = "New York City"
GRAND_CANYON_TEXT = "Grand Canyon"
TOUR_TEXT = "Tour"

DURATION_TEXT = "Duration (seconds)"
FPS_TEXT = "Frames Per Second"
TILES_LOADED_TEXT = "Tiles Loaded"

class CesiumPerformanceWindow(ui.Window):
    WINDOW_NAME = "Cesium Performance Testing"
    MENU_PATH = f"Window/Cesium/{WINDOW_NAME}"

    def __init__(self, **kwargs):
        super().__init__(CesiumPerformanceWindow.WINDOW_NAME, **kwargs)

        self._logger = logging.getLogger(__name__)

        self._random_colors_checkbox_model = ui.SimpleBoolModel(False)
        self._forbid_holes_checkbox_model = ui.SimpleBoolModel(False)
        self._frustum_culling_checkbox_model = ui.SimpleBoolModel(False)

        self._duration_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)


        self.frame.set_build_fn(self._build_fn)

    def destroy(self) -> None:
        super().destroy()

    def _build_fn(self):
        with ui.VStack(spacing=10):
            with ui.VStack(spacing=4):
                with ui.HStack(height=16):
                    ui.Label("Options", height=0)
                    ui.Spacer()

                for label, model in [
                    (RANDOM_COLORS_TEXT, self._random_colors_checkbox_model),
                    (FORBID_HOLES_TEXT, self._forbid_holes_checkbox_model),
                    (FRUSTUM_CULLING_TEXT, self._frustum_culling_checkbox_model),
                ]:
                    with ui.HStack(height=0):
                        ui.Label(label, height=0)
                        ui.CheckBox(model)

            with ui.VStack(spacing=0):
                ui.Label("Scenarios", height=16)

                for label, callback in [
                    (NEW_YORK_CITY_TEXT, self._view_new_york_city),
                    (GRAND_CANYON_TEXT, self._view_grand_canyon),
                    (TOUR_TEXT, self._view_tour),
                ]:
                    ui.Button(label, height=20, clicked_fn=callback)

            with ui.VStack(spacing=4):
                with ui.HStack(height=16):
                    ui.Label("Stats", height=0)
                    ui.Spacer()

                for label, model in [
                    (DURATION_TEXT, self._duration_model),
                    (FPS_TEXT, self._duration_model),
                    (TILES_LOADED_TEXT, self._duration_model),
                ]:
                    with ui.HStack(height=0):
                        ui.Label(label, height=0)
                        ui.StringField(model=model, height=0, read_only=True)

    def _view_new_york_city(self):
        print("View New York City")

    def _view_grand_canyon(self):
        print("View Grand Canyon")

    def _view_tour(self):
        print("Tour")

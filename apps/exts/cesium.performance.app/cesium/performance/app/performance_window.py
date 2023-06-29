import logging
import carb.events
import omni.kit.app as app
import omni.ui as ui
from cesium.omniverse.bindings import ICesiumOmniverseInterface
from cesium.omniverse.ui.models.space_delimited_number_model import SpaceDelimitedNumberModel

RANDOM_COLORS_TEXT = "Random colors"
FORBID_HOLES_TEXT = "Forbid holes"
FRUSTUM_CULLING_TEXT = "Frustum culling"
TRACING_ENABLED_TEXT = "Tracing enabled"
MAIN_THREAD_LOADING_TIME_LIMIT_TEXT = "Main thread loading time limit (ms)"

NEW_YORK_CITY_TEXT = "New York City"
PARIS_TEXT = "Paris"
GRAND_CANYON_TEXT = "Grand Canyon"
TOUR_TEXT = "Tour"

NEW_YORK_CITY_GOOGLE_TEXT = "New York City (Google)"
PARIS_GOOGLE_TEXT = "Paris (Google)"
GRAND_CANYON_GOOGLE_TEXT = "Grand Canyon (Google)"
TOUR_GOOGLE_TEXT = "Tour (Google)"

DURATION_TEXT = "Duration (seconds)"


class CesiumPerformanceWindow(ui.Window):
    WINDOW_NAME = "Cesium Performance Testing"
    MENU_PATH = f"Window/Cesium/{WINDOW_NAME}"

    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, **kwargs):
        super().__init__(CesiumPerformanceWindow.WINDOW_NAME, **kwargs)

        self._cesium_omniverse_interface = cesium_omniverse_interface
        self._logger = logging.getLogger(__name__)

        self._random_colors_checkbox_model = ui.SimpleBoolModel(False)
        self._forbid_holes_checkbox_model = ui.SimpleBoolModel(False)
        self._frustum_culling_checkbox_model = ui.SimpleBoolModel(True)
        self._main_thread_loading_time_limit_model = ui.SimpleFloatModel(0.0)

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

                with ui.HStack(height=0):
                    ui.Label(MAIN_THREAD_LOADING_TIME_LIMIT_TEXT, height=0)
                    ui.StringField(self._main_thread_loading_time_limit_model)

                with ui.HStack(height=16):
                    tracing_label = ui.Label(TRACING_ENABLED_TEXT, height=0)
                    tracing_label.set_tooltip(
                        "Enabled when the project is configured with -D CESIUM_OMNI_ENABLE_TRACING=ON"
                    )
                    enabled_string = "ON" if self._cesium_omniverse_interface.is_tracing_enabled() else "OFF"
                    ui.Label(enabled_string, height=0)

            with ui.VStack(spacing=0):
                ui.Label("Scenarios", height=16)

                for label, callback in [
                    (NEW_YORK_CITY_TEXT, self._view_new_york_city),
                    (PARIS_TEXT, self._view_paris),
                    (GRAND_CANYON_TEXT, self._view_grand_canyon),
                    (TOUR_TEXT, self._view_tour),
                    (NEW_YORK_CITY_GOOGLE_TEXT, self._view_new_york_city_google),
                    (PARIS_GOOGLE_TEXT, self._view_paris_google),
                    (GRAND_CANYON_GOOGLE_TEXT, self._view_grand_canyon_google),
                    (TOUR_GOOGLE_TEXT, self._view_tour_google),
                ]:
                    ui.Button(label, height=20, clicked_fn=callback)

            with ui.VStack(spacing=4):
                with ui.HStack(height=16):
                    ui.Label("Stats", height=0)
                    ui.Spacer()

                for label, model in [
                    (DURATION_TEXT, self._duration_model),
                ]:
                    with ui.HStack(height=0):
                        ui.Label(label, height=0)
                        ui.StringField(model=model, height=0, read_only=True)

            with ui.VStack(spacing=0):
                ui.Button("Stop", height=16, clicked_fn=self._stop)

    def _view_new_york_city(self):
        bus = app.get_app().get_message_bus_event_stream()
        view_new_york_city_event = carb.events.type_from_string("cesium.performance.VIEW_NEW_YORK_CITY")
        bus.push(view_new_york_city_event)

    def _view_paris(self):
        bus = app.get_app().get_message_bus_event_stream()
        view_paris_event = carb.events.type_from_string("cesium.performance.VIEW_PARIS")
        bus.push(view_paris_event)

    def _view_grand_canyon(self):
        bus = app.get_app().get_message_bus_event_stream()
        view_grand_canyon_event = carb.events.type_from_string("cesium.performance.VIEW_GRAND_CANYON")
        bus.push(view_grand_canyon_event)

    def _view_tour(self):
        bus = app.get_app().get_message_bus_event_stream()
        view_tour_event = carb.events.type_from_string("cesium.performance.VIEW_TOUR")
        bus.push(view_tour_event)

    def _view_new_york_city_google(self):
        bus = app.get_app().get_message_bus_event_stream()
        view_new_york_city_google_event = carb.events.type_from_string("cesium.performance.VIEW_NEW_YORK_CITY_GOOGLE")
        bus.push(view_new_york_city_google_event)

    def _view_paris_google(self):
        bus = app.get_app().get_message_bus_event_stream()
        view_paris_google_event = carb.events.type_from_string("cesium.performance.VIEW_PARIS_GOOGLE")
        bus.push(view_paris_google_event)

    def _view_grand_canyon_google(self):
        bus = app.get_app().get_message_bus_event_stream()
        view_grand_canyon_google_event = carb.events.type_from_string("cesium.performance.VIEW_GRAND_CANYON_GOOGLE")
        bus.push(view_grand_canyon_google_event)

    def _view_tour_google(self):
        bus = app.get_app().get_message_bus_event_stream()
        view_tour_google_event = carb.events.type_from_string("cesium.performance.VIEW_TOUR_GOOGLE")
        bus.push(view_tour_google_event)

    def _stop(self):
        bus = app.get_app().get_message_bus_event_stream()
        stop_event = carb.events.type_from_string("cesium.performance.STOP")
        bus.push(stop_event)

    def get_random_colors(self) -> bool:
        return self._random_colors_checkbox_model.get_value_as_bool()

    def get_forbid_holes(self) -> bool:
        return self._forbid_holes_checkbox_model.get_value_as_bool()

    def get_frustum_culling(self) -> bool:
        return self._frustum_culling_checkbox_model.get_value_as_bool()

    def get_main_thread_loading_time_limit_model(self) -> float:
        return self._main_thread_loading_time_limit_model.get_value_as_float()

    def set_duration(self, duration: float):
        self._duration_model.set_value(duration)

import logging
import omni.ui as ui
import carb.events
import omni.kit.app as app
from typing import List
from cesium.omniverse.ui.models.space_delimited_number_model import SpaceDelimitedNumberModel
from cesium.omniverse.utils.cesium_interface import CesiumInterfaceManager
from datetime import datetime
from cesium.omniverse.usdUtils import get_tileset_paths


class CesiumLoadTimerWindow(ui.Window):
    WINDOW_NAME = "Cesium Load Timer"

    _logger: logging.Logger
    _last_tiles_loading_worker = 0

    def __init__(self, **kwargs):
        super().__init__(CesiumLoadTimerWindow.WINDOW_NAME, **kwargs)

        self._logger = logging.getLogger(__name__)

        self._timer_active = False
        self._tiles_loading_worker_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._load_time_seconds_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._past_results_model: ui.SimpleStringModel = ui.SimpleStringModel("")

        self._subscriptions: List[carb.events.ISubscription] = []
        self._setup_subscriptions()

        # Set the function that is called to build widgets when the window is visible
        self.frame.set_build_fn(self._build_fn)

        self.set_visibility_changed_fn(self._visibility_changed_fn)

    def destroy(self):
        self._remove_subscriptions()
        super().destroy()

    def __del__(self):
        self.destroy()

    def _visibility_changed_fn(self, visible):
        if not visible:
            self._remove_subscriptions()

    def _setup_subscriptions(self):
        update_stream = app.get_app().get_update_event_stream()
        self._subscriptions.append(
            update_stream.create_subscription_to_pop(self._on_update_frame, name="on_update_frame")
        )

    def _remove_subscriptions(self):
        for subscription in self._subscriptions:
            subscription.unsubscribe()
        self._subscriptions.clear()

    def _on_update_frame(self, _e: carb.events.IEvent):
        if not self.visible or not self._timer_active:
            return

        with CesiumInterfaceManager() as interface:
            render_statistics = interface.get_render_statistics()

            # Loading worker count has changed from last frame, so update the timer
            if render_statistics.tiles_loading_worker != self._last_tiles_loading_worker:

                # Register a new end-time and calculate the total elapsed time in seconds
                self._end_time = datetime.now()
                time_elapsed = self._end_time - self._start_time
                self._load_time_seconds_model.set_value(time_elapsed.total_seconds())

            # If 30 sucessive frames with zero tiles loading occurs, we assume loading has finished
            if render_statistics.tiles_loading_worker == 0:
                self._zero_counter += 1
                if self._zero_counter >= 30:
                    self._end_load_timer()  # Cancel the timer after 30 successful 0 tile frames
            else:
                self._zero_counter = 0

            # Store the number of tile workers for use in the next update cycle
            self._last_tiles_loading_worker = render_statistics.tiles_loading_worker
            self._tiles_loading_worker_model.set_value(render_statistics.tiles_loading_worker)

    def _start_load_timer(self):
        self._start_time = datetime.now()
        self._end_time = datetime.now()
        self._timer_active = True
        self._zero_counter = 0
        with CesiumInterfaceManager() as interface:
            tileset_paths = get_tileset_paths()

            for tileset_path in tileset_paths:
                interface.reload_tileset(tileset_path)

    def _end_load_timer(self):
        self._timer_active = False
        result_str = f"{self._load_time_seconds_model}\n" + self._past_results_model.get_value_as_string()
        self._past_results_model.set_value(result_str)

    @staticmethod
    def create_window():
        return CesiumLoadTimerWindow(width=300, height=370)

    def _build_fn(self):
        """Builds out the UI"""

        with ui.VStack(spacing=4):

            ui.Label(
                "This tool records the amount of time taken to reload all tilesets in the stage",
                word_wrap=True,
            )

            def reload_all_tilesets():
                self._start_load_timer()

            ui.Button("Reload all Tilesets", height=20, clicked_fn=reload_all_tilesets)

            ui.Label(
                "The timer automatically completes when no tiles are queued to load for 30 successive frames",
                word_wrap=True,
            )

            for label, model in [
                ("Tiles loading (worker)", self._tiles_loading_worker_model),
                ("Load time (s)", self._load_time_seconds_model),
            ]:

                with ui.HStack(height=0):
                    ui.Label(label, height=0)
                    ui.StringField(model=model, height=0, read_only=True)

            ui.Label("Past results:", height=0)
            ui.StringField(model=self._past_results_model, multiline=True, height=150)

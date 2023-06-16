import logging
import carb.events
import omni.kit.app as app
import omni.ui as ui
from typing import List
from ..bindings import ICesiumOmniverseInterface
from .models.space_delimited_number_model import SpaceDelimitedNumberModel
from .models.human_readable_bytes_model import HumanReadableBytesModel

MATERIALS_CAPACITY_TEXT = "Materials capacity"
MATERIALS_LOADED_TEXT = "Materials loaded"
GEOMETRIES_CAPACITY_TEXT = "Geometries capacity"
GEOMETRIES_LOADED_TEXT = "Geometries loaded"
GEOMETRIES_RENDERED_TEXT = "Geometries rendered"
TRIANGLES_LOADED_TEXT = "Triangles loaded"
TRIANGLES_RENDERED_TEXT = "Triangles rendered"
TILESET_CACHED_BYTES_TEXT = "Tileset cached bytes"
TILESET_CACHED_BYTES_HUMAN_READABLE_TEXT = "Tileset cached bytes (Human-readable)"
TILES_VISITED_TEXT = "Tiles visited"
CULLED_TILES_VISITED_TEXT = "Culled tiles visited"
TILES_RENDERED_TEXT = "Tiles rendered"
TILES_CULLED_TEXT = "Tiles culled"
MAX_DEPTH_VISITED_TEXT = "Max depth visited"
TILES_LOADING_WORKER_TEXT = "Tiles loading (worker)"
TILES_LOADING_MAIN_TEXT = "Tiles loading (main)"
TILES_LOADED_TEXT = "Tiles loaded"


class CesiumOmniverseStatisticsWidget(ui.Frame):
    """
    Widget that displays statistics about the scene.
    """

    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, **kwargs):
        super().__init__(build_fn=self._build_fn, **kwargs)

        self._logger = logging.getLogger(__name__)

        self._cesium_omniverse_interface = cesium_omniverse_interface

        self._materials_capacity_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._materials_loaded_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._geometries_capacity_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._geometries_loaded_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._geometries_rendered_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._triangles_loaded_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._triangles_rendered_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._tileset_cached_bytes_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._tileset_cached_bytes_human_readable_model: HumanReadableBytesModel = HumanReadableBytesModel(0)
        self._tiles_visited_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._culled_tiles_visited_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._tiles_rendered_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._tiles_culled_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._max_depth_visited_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._tiles_loading_worker_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._tiles_loading_main_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._tiles_loaded_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)

        self._subscriptions: List[carb.events.ISubscription] = []
        self._setup_subscriptions()

    def __del__(self):
        self.destroy()

    def destroy(self):
        for subscription in self._subscriptions:
            subscription.unsubscribe()
        self._subscriptions.clear()

        super().destroy()

    def _setup_subscriptions(self):
        update_stream = app.get_app().get_update_event_stream()
        self._subscriptions.append(
            update_stream.create_subscription_to_pop(self._on_update_frame, name="on_update_frame")
        )

    def _on_update_frame(self, _e: carb.events.IEvent):
        if not self.visible:
            return

        render_statistics = self._cesium_omniverse_interface.get_render_statistics()
        self._materials_capacity_model.set_value(render_statistics.materials_capacity)
        self._materials_loaded_model.set_value(render_statistics.materials_loaded)
        self._geometries_capacity_model.set_value(render_statistics.geometries_capacity)
        self._geometries_loaded_model.set_value(render_statistics.geometries_loaded)
        self._geometries_rendered_model.set_value(render_statistics.geometries_rendered)
        self._triangles_loaded_model.set_value(render_statistics.triangles_loaded)
        self._triangles_rendered_model.set_value(render_statistics.triangles_rendered)
        self._tileset_cached_bytes_model.set_value(render_statistics.tileset_cached_bytes)
        self._tileset_cached_bytes_human_readable_model.set_value(render_statistics.tileset_cached_bytes)
        self._tiles_visited_model.set_value(render_statistics.tiles_visited)
        self._culled_tiles_visited_model.set_value(render_statistics.culled_tiles_visited)
        self._tiles_rendered_model.set_value(render_statistics.tiles_rendered)
        self._tiles_culled_model.set_value(render_statistics.tiles_culled)
        self._max_depth_visited_model.set_value(render_statistics.max_depth_visited)
        self._tiles_loading_worker_model.set_value(render_statistics.tiles_loading_worker)
        self._tiles_loading_main_model.set_value(render_statistics.tiles_loading_main)
        self._tiles_loaded_model.set_value(render_statistics.tiles_loaded)

    def _build_fn(self):
        """Builds all UI components."""

        with ui.VStack(spacing=4):
            with ui.HStack(height=16):
                ui.Label("Statistics", height=0)
                ui.Spacer()

            for label, model in [
                (MATERIALS_CAPACITY_TEXT, self._materials_capacity_model),
                (MATERIALS_LOADED_TEXT, self._materials_loaded_model),
                (GEOMETRIES_CAPACITY_TEXT, self._geometries_capacity_model),
                (GEOMETRIES_LOADED_TEXT, self._geometries_loaded_model),
                (GEOMETRIES_RENDERED_TEXT, self._geometries_rendered_model),
                (TRIANGLES_LOADED_TEXT, self._triangles_loaded_model),
                (TRIANGLES_RENDERED_TEXT, self._triangles_rendered_model),
                (TILESET_CACHED_BYTES_TEXT, self._tileset_cached_bytes_model),
                (TILESET_CACHED_BYTES_HUMAN_READABLE_TEXT, self._tileset_cached_bytes_human_readable_model),
                (TILES_VISITED_TEXT, self._tiles_visited_model),
                (CULLED_TILES_VISITED_TEXT, self._culled_tiles_visited_model),
                (TILES_RENDERED_TEXT, self._tiles_rendered_model),
                (TILES_CULLED_TEXT, self._tiles_culled_model),
                (MAX_DEPTH_VISITED_TEXT, self._max_depth_visited_model),
                (TILES_LOADING_WORKER_TEXT, self._tiles_loading_worker_model),
                (TILES_LOADING_MAIN_TEXT, self._tiles_loading_main_model),
                (TILES_LOADED_TEXT, self._tiles_loaded_model),
            ]:
                with ui.HStack(height=0):
                    ui.Label(label, height=0)
                    ui.StringField(model=model, height=0, read_only=True)

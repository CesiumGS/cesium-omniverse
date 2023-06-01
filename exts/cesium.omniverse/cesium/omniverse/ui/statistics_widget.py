import logging
import carb.events
import omni.kit.app as app
import omni.ui as ui
from typing import List
from ..bindings import ICesiumOmniverseInterface
from .models.space_delimited_number_model import SpaceDelimitedNumberModel
from .models.human_readable_bytes_model import HumanReadableBytesModel

MATERIALS_LOADED_TEXT = "Materials loaded"
GEOMETRIES_LOADED_TEXT = "Geometries loaded"
GEOMETRIES_VISIBLE_TEXT = "Geometries visible"
TRIANGLES_LOADED_TEXT = "Triangles loaded"
TRIANGLES_VISIBLE_TEXT = "Triangles visible"
TILESET_CACHED_BYTES_TEXT = "Tileset cached bytes"
TILESET_CACHED_BYTES_HUMAN_READABLE_TEXT = "Tileset cached bytes (Human-readable)"


class CesiumOmniverseStatisticsWidget(ui.Frame):
    """
    Widget that displays statistics about the scene.
    """

    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, **kwargs):
        super().__init__(build_fn=self._build_fn, **kwargs)

        self._logger = logging.getLogger(__name__)

        self._cesium_omniverse_interface = cesium_omniverse_interface

        self._materials_loaded_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._geometries_loaded_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._geometries_visible_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._triangles_loaded_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._triangles_visible_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._tileset_cached_bytes_model: SpaceDelimitedNumberModel = SpaceDelimitedNumberModel(0)
        self._tileset_cached_bytes_human_readable_model: HumanReadableBytesModel = HumanReadableBytesModel(0)

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
        self._materials_loaded_model.set_value(render_statistics.number_of_materials_loaded)
        self._geometries_loaded_model.set_value(render_statistics.number_of_geometries_loaded)
        self._geometries_visible_model.set_value(render_statistics.number_of_geometries_visible)
        self._triangles_loaded_model.set_value(render_statistics.number_of_triangles_loaded)
        self._triangles_visible_model.set_value(render_statistics.number_of_triangles_visible)
        self._tileset_cached_bytes_model.set_value(render_statistics.tileset_cached_bytes)
        self._tileset_cached_bytes_human_readable_model.set_value(render_statistics.tileset_cached_bytes)

    def _build_fn(self):
        """Builds all UI components."""

        with ui.VStack(spacing=4):
            with ui.HStack(height=16):
                ui.Label("Statistics", height=0)
                ui.Spacer()

            for label, model in [
                (MATERIALS_LOADED_TEXT, self._materials_loaded_model),
                (GEOMETRIES_LOADED_TEXT, self._geometries_loaded_model),
                (GEOMETRIES_VISIBLE_TEXT, self._geometries_visible_model),
                (TRIANGLES_LOADED_TEXT, self._triangles_loaded_model),
                (TRIANGLES_VISIBLE_TEXT, self._triangles_visible_model),
                (TILESET_CACHED_BYTES_TEXT, self._tileset_cached_bytes_model),
                (TILESET_CACHED_BYTES_HUMAN_READABLE_TEXT, self._tileset_cached_bytes_human_readable_model),
            ]:
                with ui.HStack(height=0):
                    ui.Label(label, height=0)
                    ui.StringField(model=model, height=0, read_only=True)

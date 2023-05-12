import carb.events
import omni.kit.app as app
import omni.ui as ui
from typing import List
from ..bindings import ICesiumOmniverseInterface

NUMBER_OF_MATERIALS_LOADED_TEXT = "Number of materials loaded: {0}"
NUMBER_OF_GEOMETRIES_LOADED_TEXT = "Number of geometries loaded: {0}"
NUMBER_OF_GEOMETRIES_VISIBLE_TEXT = "Number of geometries visible: {0}"
NUMBER_OF_TRIANGLES_LOADED_TEXT = "Number of triangles loaded: {0}"
NUMBER_OF_TRIANGLES_VISIBLE_TEXT = "Number of triangles visible: {0}"
TILESET_CACHED_BYTES_TEXT = "Tileset cached bytes: {0}"


class CesiumOmniverseStatisticsWidget(ui.Frame):
    """
    Widget that displays statistics about the scene.
    """

    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, **kwargs):
        super().__init__(build_fn=self._build_fn, **kwargs)

        self._cesium_omniverse_interface = cesium_omniverse_interface

        self._statistics_number_of_materials_loaded_field: ui.SimpleStringModel = ui.SimpleStringModel("")
        self._statistics_number_of_geometries_loaded_field: ui.SimpleStringModel = ui.SimpleStringModel("")
        self._statistics_number_of_geometries_visible_field: ui.SimpleStringModel = ui.SimpleStringModel("")
        self._statistics_number_of_triangles_loaded_field: ui.SimpleStringModel = ui.SimpleStringModel("")
        self._statistics_number_of_triangles_visible_field: ui.SimpleStringModel = ui.SimpleStringModel("")
        self._statistics_tileset_cached_bytes_field: ui.SimpleStringModel = ui.SimpleStringModel("")

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
        fabric_statistics = render_statistics.fabric_statistics
        self._statistics_number_of_materials_loaded_field.set_value(
            NUMBER_OF_MATERIALS_LOADED_TEXT.format(fabric_statistics.number_of_materials_loaded)
        )
        self._statistics_number_of_geometries_loaded_field.set_value(
            NUMBER_OF_GEOMETRIES_LOADED_TEXT.format(fabric_statistics.number_of_geometries_loaded)
        )
        self._statistics_number_of_geometries_visible_field.set_value(
            NUMBER_OF_GEOMETRIES_VISIBLE_TEXT.format(fabric_statistics.number_of_geometries_visible)
        )
        self._statistics_number_of_triangles_loaded_field.set_value(
            NUMBER_OF_TRIANGLES_LOADED_TEXT.format(fabric_statistics.number_of_triangles_loaded)
        )
        self._statistics_number_of_triangles_visible_field.set_value(
            NUMBER_OF_TRIANGLES_VISIBLE_TEXT.format(fabric_statistics.number_of_triangles_visible)
        )
        self._statistics_tileset_cached_bytes_field.set_value(
            TILESET_CACHED_BYTES_TEXT.format(render_statistics.tileset_cached_bytes)
        )

    def _build_fn(self):
        """Builds all UI components."""

        with ui.VStack():
            ui.Label("Statistics", height=0)
            ui.StringField(self._statistics_number_of_materials_loaded_field, height=0, read_only=True)
            ui.StringField(self._statistics_number_of_geometries_loaded_field, height=0, read_only=True)
            ui.StringField(self._statistics_number_of_geometries_visible_field, height=0, read_only=True)
            ui.StringField(self._statistics_number_of_triangles_loaded_field, height=0, read_only=True)
            ui.StringField(self._statistics_number_of_triangles_visible_field, height=0, read_only=True)
            ui.StringField(self._statistics_tileset_cached_bytes_field, height=0, read_only=True)

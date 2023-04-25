import carb.events
import omni.kit.app as app
import omni.ui as ui
from typing import List, Optional
from ..bindings import ICesiumOmniverseInterface


class CesiumOmniverseStatisticsWidget(ui.Frame):
    """
    Widget that displays statistics about the scene.
    """

    NUMBER_OF_MATERIALS_LOADED_TEXT = "Number of materials loaded: {0}"
    NUMBER_OF_GEOMETRIES_LOADED_TEXT = "Number of geometries loaded: {0}"
    NUMBER_OF_GEOMETRIES_VISIBLE_TEXT = "Number of geometries visible: {0}"

    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, **kwargs):
        super().__init__(build_fn=self._build_fn, **kwargs)

        self._cesium_omniverse_interface = cesium_omniverse_interface

        self._statistics_number_of_materials_loaded_field: Optional[ui.SimpleStringModel] = None
        self._statistics_number_of_geometries_loaded_field: Optional[ui.SimpleStringModel] = None
        self._statistics_number_of_geometries_visible_field: Optional[ui.SimpleStringModel] = None

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

        fabric_statistics = self._cesium_omniverse_interface.get_fabric_statistics()
        self._statistics_number_of_materials_loaded_field.set_value(
            CesiumOmniverseStatisticsWidget.NUMBER_OF_MATERIALS_LOADED_TEXT.format(
                fabric_statistics.number_of_materials_loaded
            )
        )
        self._statistics_number_of_geometries_loaded_field.set_value(
            CesiumOmniverseStatisticsWidget.NUMBER_OF_GEOMETRIES_LOADED_TEXT.format(
                fabric_statistics.number_of_geometries_loaded
            )
        )
        self._statistics_number_of_geometries_visible_field.set_value(
            CesiumOmniverseStatisticsWidget.NUMBER_OF_GEOMETRIES_VISIBLE_TEXT.format(
                fabric_statistics.number_of_geometries_visible
            )
        )

    def _build_fn(self):
        """Builds all UI components."""

        with ui.VStack():
            ui.Label("Statistics", height=0)
            self._statistics_number_of_materials_loaded_field = ui.SimpleStringModel("")
            self._statistics_number_of_geometries_loaded_field = ui.SimpleStringModel("")
            self._statistics_number_of_geometries_visible_field = ui.SimpleStringModel("")
            ui.StringField(self._statistics_number_of_materials_loaded_field, height=0, read_only=True)
            ui.StringField(self._statistics_number_of_geometries_loaded_field, height=0, read_only=True)
            ui.StringField(self._statistics_number_of_geometries_visible_field, height=0, read_only=True)

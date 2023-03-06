import logging
import carb.events
import omni.kit.app as app
import omni.ui as ui
from typing import List, Optional
from ..bindings import ICesiumOmniverseInterface
from ..models import AssetToAdd
from .styles import CesiumOmniverseUiStyles

LABEL_HEIGHT = 24
BUTTON_HEIGHT = 40


class CesiumOmniverseQuickAddWidget(ui.Frame):
    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, **kwargs):
        self._logger = logging.getLogger(__name__)
        self._cesium_omniverse_interface = cesium_omniverse_interface

        self._ion_quick_add_frame: Optional[ui.Frame] = None

        self._subscriptions: List[carb.events.ISubscription] = []
        self._setup_subscriptions()

        super().__init__(build_fn=self._build_ui, **kwargs)

    def destroy(self) -> None:
        for subscription in self._subscriptions:
            subscription.unsubscribe()
        self._subscriptions.clear()

    def _setup_subscriptions(self):
        update_stream = app.get_app().get_update_event_stream()
        self._subscriptions.append(
            update_stream.create_subscription_to_pop(self._on_update_frame, name="on_update_frame")
        )

    def _on_update_frame(self, _: carb.events.IEvent):
        if self._ion_quick_add_frame is None:
            return

        session = self._cesium_omniverse_interface.get_session()

        if session is not None:
            self._ion_quick_add_frame.visible = session.is_connected()

    @staticmethod
    def _add_blank_button_clicked():
        asset_to_add = AssetToAdd("Cesium Tileset", 0)
        add_blank_asset_event = carb.events.type_from_string("cesium.omniverse.ADD_BLANK_ASSET")
        app.get_app().get_message_bus_event_stream().push(add_blank_asset_event, payload=asset_to_add.to_dict())

    def _cwt_bing_maps_button_clicked(self):
        self._add_ion_assets(AssetToAdd("Cesium World Terrain", 1, "Bing Maps Aerial imagery", 2))

    def _cwt_bing_maps_labels_button_clicked(self):
        self._add_ion_assets(AssetToAdd("Cesium World Terrain", 1, "Bing Maps Aerial with Labels imagery", 3))

    def _cwt_bing_maps_roads_button_clicked(self):
        self._add_ion_assets(AssetToAdd("Cesium World Terrain", 1, "Bing Maps Road imagery", 4))

    def _cwt_sentinel_button_clicked(self):
        self._add_ion_assets(AssetToAdd("Cesium World Terrain", 1, "Sentinel-2 imagery", 3954))

    def _cesium_osm_buildings_clicked(self):
        self._add_ion_assets(AssetToAdd("Cesium OSM Buildings", 96188))

    @staticmethod
    def _add_ion_assets(asset_to_add: AssetToAdd):
        add_asset_event = carb.events.type_from_string("cesium.omniverse.ADD_ION_ASSET")
        app.get_app().get_message_bus_event_stream().push(add_asset_event, payload=asset_to_add.to_dict())

    def _build_ui(self):
        with self:
            with ui.VStack(spacing=10):
                with ui.VStack(spacing=5):
                    ui.Label(
                        "Quick Add Basic Assets",
                        style=CesiumOmniverseUiStyles.quick_add_section_label,
                        height=LABEL_HEIGHT,
                    )
                    ui.Button(
                        "Blank 3D Tiles Tileset",
                        style=CesiumOmniverseUiStyles.quick_add_button,
                        clicked_fn=self._add_blank_button_clicked,
                        height=BUTTON_HEIGHT,
                    )
                self._ion_quick_add_frame = ui.Frame(visible=False, height=0)
                with self._ion_quick_add_frame:
                    with ui.VStack(spacing=5):
                        ui.Label(
                            "Quick Add Cesium ion Assets",
                            style=CesiumOmniverseUiStyles.quick_add_section_label,
                            height=LABEL_HEIGHT,
                        )
                        ui.Button(
                            "Cesium World Terrain + Bing Maps Aerial imagery",
                            style=CesiumOmniverseUiStyles.quick_add_button,
                            height=BUTTON_HEIGHT,
                            clicked_fn=self._cwt_bing_maps_button_clicked,
                        )
                        ui.Button(
                            "Cesium World Terrain + Bing Maps with Labels imagery",
                            style=CesiumOmniverseUiStyles.quick_add_button,
                            height=BUTTON_HEIGHT,
                            clicked_fn=self._cwt_bing_maps_labels_button_clicked,
                        )
                        ui.Button(
                            "Cesium World Terrain + Bing Maps Road imagery",
                            style=CesiumOmniverseUiStyles.quick_add_button,
                            height=BUTTON_HEIGHT,
                            clicked_fn=self._cwt_bing_maps_roads_button_clicked,
                        )
                        ui.Button(
                            "Cesium World Terrain + Sentinel-2 imagery",
                            style=CesiumOmniverseUiStyles.quick_add_button,
                            height=BUTTON_HEIGHT,
                            clicked_fn=self._cwt_sentinel_button_clicked,
                        )
                        ui.Button(
                            "Cesium OSM Buildings",
                            style=CesiumOmniverseUiStyles.quick_add_button,
                            height=BUTTON_HEIGHT,
                            clicked_fn=self._cesium_osm_buildings_clicked,
                        )

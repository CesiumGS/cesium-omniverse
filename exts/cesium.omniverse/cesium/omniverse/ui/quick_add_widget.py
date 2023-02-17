import logging
import carb.events
import omni.kit.app as app
import omni.ui as ui
import omni.usd
import omni.kit.commands as omni_commands
from omni.kit.viewport.utility import get_active_viewport_camera_path
from pxr import Gf, Sdf
from typing import List, Optional
from ..bindings import ICesiumOmniverseInterface
from .styles import CesiumOmniverseUiStyles

LABEL_HEIGHT = 24
BUTTON_HEIGHT = 40
DEFAULT_GEOREFERENCE_LATITUDE = 39.9501464
DEFAULT_GEOREFERENCE_LONGITUDE = -75.1564977
DEFAULT_GEOREFERENCE_HEIGHT = 150.0


class AssetToAdd:
    def __init__(self, tileset_name: str, tileset_ion_id: int, imagery_name: Optional[str] = None,
                 imagery_ion_id: Optional[int] = None):
        self.tileset_name = tileset_name
        self.tileset_ion_id = tileset_ion_id
        self.imagery_name = imagery_name
        self.imagery_ion_id = imagery_ion_id


class CesiumOmniverseQuickAddWidget(ui.Frame):
    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, **kwargs):
        self._logger = logging.getLogger(__name__)
        self._cesium_omniverse_interface = cesium_omniverse_interface

        self._ion_quick_add_frame: Optional[ui.Frame] = None

        self._subscriptions: List[carb.events.ISubscription] = []
        self._setup_subscriptions()

        self._assets_to_add_after_token_set: List[AssetToAdd] = []
        self._adding_assets = False

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

        bus = app.get_app().get_message_bus_event_stream()
        token_set_event = carb.events.type_from_string("cesium.omniverse.SET_DEFAULT_TOKEN_SUCCESS")
        self._subscriptions.append(
            bus.create_subscription_to_pop_by_type(token_set_event, self._on_token_set)
        )

    def _on_update_frame(self, _: carb.events.IEvent):
        if self._ion_quick_add_frame is None:
            return

        session = self._cesium_omniverse_interface.get_session()

        if session is not None:
            self._ion_quick_add_frame.visible = session.is_connected()

    def _on_token_set(self, _: carb.events.IEvent):
        if self._adding_assets:
            return

        self._adding_assets = True

        for asset in self._assets_to_add_after_token_set:
            self._add_ion_assets(asset)
        self._assets_to_add_after_token_set.clear()

        self._adding_assets = False

    def _extend_camera_far_plane(self):
        # Set the Far Plane to a very high number so the globe is visible on zoom extents
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(
            get_active_viewport_camera_path())
        omni_commands.execute("ChangeProperty",
                              prop_path=Sdf.Path(
                                  "/OmniverseKit_Persp.clippingRange"),
                              value=Gf.Vec2f(1.0, 10000000000.0),
                              prev=camera_prim.GetAttribute("clippingRange").Get())

    def _add_blank_button_clicked(self):
        pass

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

    def _add_ion_assets(self, asset_to_add: AssetToAdd):
        session = self._cesium_omniverse_interface.get_session()

        if not session.is_connected():
            self._logger.warning("Must be logged in to add ion asset.")
            return

        if not self._cesium_omniverse_interface.is_default_token_set():
            bus = app.get_app().get_message_bus_event_stream()
            show_token_window_event = carb.events.type_from_string("cesium.omniverse.SHOW_TOKEN_WINDOW")
            bus.push(show_token_window_event)
            self._assets_to_add_after_token_set.append(asset_to_add)
            return

        # TODO: It may be better to prompt the user before brute forcing a change to their camera settings.
        self._extend_camera_far_plane()

        # TODO: Probably need a check here for bypassing setting the georeference if it is already set.
        self._cesium_omniverse_interface.set_georeference_origin(DEFAULT_GEOREFERENCE_LONGITUDE,
                                                                 DEFAULT_GEOREFERENCE_LATITUDE,
                                                                 DEFAULT_GEOREFERENCE_HEIGHT)

        if asset_to_add.imagery_name is not None and asset_to_add.imagery_ion_id is not None:
            tileset_id = self._cesium_omniverse_interface.add_tileset_and_raster_overlay(asset_to_add.tileset_name,
                                                                                         asset_to_add.tileset_ion_id,
                                                                                         asset_to_add.imagery_name,
                                                                                         asset_to_add.imagery_ion_id)
        else:
            tileset_id = self._cesium_omniverse_interface.add_tileset_ion(asset_to_add.tileset_name,
                                                                          asset_to_add.tileset_ion_id)

        if tileset_id == -1:
            # TODO: Open token troubleshooter.
            self._logger.warning("Error adding tileset and raster overlay to stage")

    def _build_ui(self):
        with self:
            with ui.VStack(spacing=10):
                with ui.VStack(spacing=5):
                    ui.Label("Quick Add Basic Assets", style=CesiumOmniverseUiStyles.quick_add_section_label,
                             height=LABEL_HEIGHT)
                    ui.Button("Blank 3D Tiles Tileset", style=CesiumOmniverseUiStyles.quick_add_button,
                              clicked_fn=self._add_blank_button_clicked, height=BUTTON_HEIGHT)
                self._ion_quick_add_frame = ui.Frame(visible=False, height=0)
                with self._ion_quick_add_frame:
                    with ui.VStack(spacing=5):
                        ui.Label("Quick Add Cesium ion Assets", style=CesiumOmniverseUiStyles.quick_add_section_label,
                                 height=LABEL_HEIGHT)
                        ui.Button("Cesium World Terrain + Bing Maps Aerial imagery",
                                  style=CesiumOmniverseUiStyles.quick_add_button, height=BUTTON_HEIGHT,
                                  clicked_fn=self._cwt_bing_maps_button_clicked)
                        ui.Button("Cesium World Terrain + Bing Maps with Labels imagery",
                                  style=CesiumOmniverseUiStyles.quick_add_button, height=BUTTON_HEIGHT,
                                  clicked_fn=self._cwt_bing_maps_labels_button_clicked)
                        ui.Button("Cesium World Terrain + Bing Maps Road imagery",
                                  style=CesiumOmniverseUiStyles.quick_add_button, height=BUTTON_HEIGHT,
                                  clicked_fn=self._cwt_bing_maps_roads_button_clicked)
                        ui.Button("Cesium World Terrain + Sentinel-2 imagery",
                                  style=CesiumOmniverseUiStyles.quick_add_button, height=BUTTON_HEIGHT,
                                  clicked_fn=self._cwt_sentinel_button_clicked)
                        ui.Button("Cesium OSM Buildings", style=CesiumOmniverseUiStyles.quick_add_button,
                                  height=BUTTON_HEIGHT,
                                  clicked_fn=self._cesium_osm_buildings_clicked)

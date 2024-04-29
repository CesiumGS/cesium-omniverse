from typing import Optional
import carb.events
import omni.kit.app as app
import omni.ui as ui
import omni.usd as usd
from ..bindings import ICesiumOmniverseInterface
from .models import IonAssetItem
from ..models import AssetToAdd, RasterOverlayToAdd
from .styles import CesiumOmniverseUiStyles
from ..usdUtils import is_tileset, get_tileset_paths


class CesiumAssetDetailsWidget(ui.ScrollingFrame):
    def __init__(
        self, cesium_omniverse_interface: ICesiumOmniverseInterface, asset: Optional[IonAssetItem] = None, **kwargs
    ):
        super().__init__(**kwargs)

        self._cesium_omniverse_interface = cesium_omniverse_interface

        self.style = CesiumOmniverseUiStyles.asset_detail_frame

        self._name = asset.name.as_string if asset else ""
        self._id = asset.id.as_int if asset else 0
        self._description = asset.description.as_string if asset else ""
        self._attribution = asset.attribution.as_string if asset else ""
        self._asset_type = asset.type.as_string if asset else ""

        self._name_label: Optional[ui.Label] = None
        self._id_label: Optional[ui.Label] = None
        self._description_label: Optional[ui.Label] = None
        self._attribution_label: Optional[ui.Label] = None

        self.set_build_fn(self._build_fn)

    def __del__(self):
        self.destroy()

    def destroy(self) -> None:
        if self._name_label is not None:
            self._name_label.destroy()

        if self._id_label is not None:
            self._id_label.destroy()

        if self._description_label is not None:
            self._description_label.destroy()

        if self._attribution_label is not None:
            self._attribution_label.destroy()

    def update_selection(self, asset: Optional[IonAssetItem]):
        self._name = asset.name.as_string if asset else ""
        self._id = asset.id.as_int if asset else 0
        self._description = asset.description.as_string if asset else ""
        self._attribution = asset.attribution.as_string if asset else ""
        self._asset_type = asset.type.as_string if asset else ""

        self.rebuild()

    def _should_be_visible(self):
        return self._name != "" or self._id != 0 or self._description != "" or self._attribution != ""

    def _add_overlay_with_tileset(self):
        asset_to_add = AssetToAdd("Cesium World Terrain", 1, self._name, self._id)
        add_asset_event = carb.events.type_from_string("cesium.omniverse.ADD_ION_ASSET")
        app.get_app().get_message_bus_event_stream().push(add_asset_event, payload=asset_to_add.to_dict())

    def _add_tileset_button_clicked(self):
        asset_to_add = AssetToAdd(self._name, self._id)
        add_asset_event = carb.events.type_from_string("cesium.omniverse.ADD_ION_ASSET")
        app.get_app().get_message_bus_event_stream().push(add_asset_event, payload=asset_to_add.to_dict())

    def _add_raster_overlay_button_clicked(self):
        context = usd.get_context()
        selection = context.get_selection().get_selected_prim_paths()
        tileset_path: Optional[str] = None

        if len(selection) > 0 and is_tileset(selection[0]):
            tileset_path = selection[0]

        if tileset_path is None:
            all_tileset_paths = get_tileset_paths()

            if len(all_tileset_paths) > 0:
                tileset_path = all_tileset_paths[0]
            else:
                self._add_overlay_with_tileset()
                return

        raster_overlay_to_add = RasterOverlayToAdd(tileset_path, self._id, self._name)

        add_raster_overlay_event = carb.events.type_from_string("cesium.omniverse.ADD_RASTER_OVERLAY")
        app.get_app().get_message_bus_event_stream().push(
            add_raster_overlay_event, payload=raster_overlay_to_add.to_dict()
        )

    def _build_fn(self):
        with self:
            if self._should_be_visible():
                with ui.VStack(spacing=20):
                    with ui.VStack(spacing=5):
                        ui.Label(
                            self._name,
                            style=CesiumOmniverseUiStyles.asset_detail_name_label,
                            height=0,
                            word_wrap=True,
                        )
                        ui.Label(
                            f"(ID: {self._id})",
                            style=CesiumOmniverseUiStyles.asset_detail_id_label,
                            height=0,
                            word_wrap=True,
                        )
                    with ui.HStack(spacing=0, height=0):
                        ui.Spacer(height=0)
                        if self._asset_type == "3DTILES" or self._asset_type == "TERRAIN":
                            ui.Button(
                                "Add to Stage",
                                width=0,
                                height=0,
                                style=CesiumOmniverseUiStyles.blue_button_style,
                                clicked_fn=self._add_tileset_button_clicked,
                            )
                        elif self._asset_type == "IMAGERY":
                            ui.Button(
                                "Use as Terrain Tileset Base Layer",
                                width=0,
                                height=0,
                                style=CesiumOmniverseUiStyles.blue_button_style,
                                clicked_fn=self._add_raster_overlay_button_clicked,
                            )
                        else:
                            # Skipping adding a button for things we cannot add for now.
                            pass
                        ui.Spacer(height=0)
                    with ui.VStack(spacing=5):
                        ui.Label("Description", style=CesiumOmniverseUiStyles.asset_detail_header_label, height=0)
                        ui.Label(self._description, word_wrap=True, alignment=ui.Alignment.TOP, height=0)
                    with ui.VStack(spacing=5):
                        ui.Label("Attribution", style=CesiumOmniverseUiStyles.asset_detail_header_label, height=0)
                        ui.Label(self._attribution, word_wrap=True, alignment=ui.Alignment.TOP, height=0)
            else:
                ui.Spacer()

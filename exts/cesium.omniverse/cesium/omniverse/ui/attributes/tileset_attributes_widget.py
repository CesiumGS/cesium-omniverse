import logging
from omni.kit.property.usd.custom_layout_helper import CustomLayoutFrame, CustomLayoutGroup, CustomLayoutProperty
from omni.kit.property.usd.usd_property_widget import SchemaPropertiesWidget
import omni.ui as ui
from ...bindings import ICesiumOmniverseInterface
from cesium.usd.plugins.CesiumUsdSchemas import Tileset as CesiumTileset, IonServer as CesiumIonServer


class CesiumTilesetAttributesWidget(SchemaPropertiesWidget):
    def __init__(self, _cesium_omniverse_interface: ICesiumOmniverseInterface):
        super().__init__("Cesium Tileset Settings", CesiumTileset, include_inherited=False)

        self._logger = logging.getLogger(__name__)

        self._cesium_omniverse_interface = _cesium_omniverse_interface

    def clean(self):
        super().clean()

    def on_refresh_button_clicked(self):
        tileset_path = self._payload[0]
        self._cesium_omniverse_interface.reload_tileset(tileset_path.pathString)

    def _customize_props_layout(self, props):
        frame = CustomLayoutFrame(hide_extra=True)

        with frame:
            ui.Button("Refresh Tileset", clicked_fn=self.on_refresh_button_clicked)
            with CustomLayoutGroup("Credit Display"):
                CustomLayoutProperty("cesium:showCreditsOnScreen")
            with CustomLayoutGroup("Source"):
                CustomLayoutProperty("cesium:sourceType")
                CustomLayoutProperty("cesium:ionAssetId")
                CustomLayoutProperty("cesium:ionAccessToken")
                CustomLayoutProperty("cesium:ionServerBinding")
                CustomLayoutProperty("cesium:url")
            with CustomLayoutGroup("Level of Detail"):
                CustomLayoutProperty("cesium:maximumScreenSpaceError")
            with CustomLayoutGroup("Tile Loading"):
                CustomLayoutProperty("cesium:preloadAncestors")
                CustomLayoutProperty("cesium:preloadSiblings")
                CustomLayoutProperty("cesium:forbidHoles")
                CustomLayoutProperty("cesium:maximumSimultaneousTileLoads")
                CustomLayoutProperty("cesium:maximumCachedBytes")
                CustomLayoutProperty("cesium:loadingDescendantLimit")
                CustomLayoutProperty("cesium:mainThreadLoadingTimeLimit")
            with CustomLayoutGroup("Tile Culling"):
                CustomLayoutProperty("cesium:enableFrustumCulling")
                CustomLayoutProperty("cesium:enableFogCulling")
                CustomLayoutProperty("cesium:enforceCulledScreenSpaceError")
                CustomLayoutProperty("cesium:culledScreenSpaceError")
            with CustomLayoutGroup("Rendering"):
                CustomLayoutProperty("cesium:suspendUpdate")
                CustomLayoutProperty("cesium:smoothNormals")

        return frame.apply(props)

    def _filter_props_to_build(self, props):
        filtered_props = super()._filter_props_to_build(props)
        filtered_props.extend(prop for prop in props if prop.GetName() == "cesium:ionServerBinding")
        return filtered_props

    def get_additional_kwargs(self, ui_attr):
        if ui_attr.prop_name == "cesium:ionServerBinding":
            return None, {"target_picker_filter_type_list": [CesiumIonServer], "targets_limit": 1}
        return None, {"targets_limit": 0}

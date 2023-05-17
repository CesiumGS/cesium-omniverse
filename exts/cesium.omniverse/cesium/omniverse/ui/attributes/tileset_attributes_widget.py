import logging
from omni.kit.property.usd.custom_layout_helper import CustomLayoutFrame, CustomLayoutGroup, CustomLayoutProperty
from omni.kit.property.usd.usd_property_widget import SchemaPropertiesWidget
import omni.ui as ui
from ...bindings import ICesiumOmniverseInterface
from cesium.usd.plugins.CesiumUsdSchemas import TilesetAPI as CesiumTilesetAPI


class CesiumTilesetAttributesWidget(SchemaPropertiesWidget):
    def __init__(self, _cesium_omniverse_interface: ICesiumOmniverseInterface):
        super().__init__("Cesium Tileset Settings", CesiumTilesetAPI, include_inherited=False)

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
            with CustomLayoutGroup("Tile Culling"):
                CustomLayoutProperty("cesium:enableFrustumCulling")
                CustomLayoutProperty("cesium:enableFogCulling")
                CustomLayoutProperty("cesium:enforceCulledScreenSpaceError")
                CustomLayoutProperty("cesium:culledScreenSpaceError")
            with CustomLayoutGroup("Rendering"):
                CustomLayoutProperty("cesium:suspendUpdate")
                CustomLayoutProperty("cesium:smoothNormals")

        return frame.apply(props)

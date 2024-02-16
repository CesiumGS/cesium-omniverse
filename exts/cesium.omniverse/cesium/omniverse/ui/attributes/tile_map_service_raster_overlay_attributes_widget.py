import logging
from omni.kit.property.usd.custom_layout_helper import CustomLayoutFrame, CustomLayoutGroup, CustomLayoutProperty
from omni.kit.property.usd.usd_property_widget import SchemaPropertiesWidget
from cesium.usd.plugins.CesiumUsdSchemas import (
    TileMapServiceRasterOverlay as CesiumTileMapServiceRasterOverlay,
)
from .cesium_properties_widget_builder import build_slider, build_common_raster_overlay_properties


class CesiumTileMapServiceRasterOverlayAttributesWidget(SchemaPropertiesWidget):
    def __init__(self):
        super().__init__(
            "Cesium Tile Map Service Raster Overlay Settings",
            CesiumTileMapServiceRasterOverlay,
            include_inherited=True,
        )

        self._logger = logging.getLogger(__name__)

    def clean(self):
        super().clean()

    def _customize_props_layout(self, props):
        frame = CustomLayoutFrame(hide_extra=True)

        with frame:
            with CustomLayoutGroup("URL"):
                CustomLayoutProperty("cesium:url")
            with CustomLayoutGroup("Zoom Settings"):
                CustomLayoutProperty(
                    "cesium:specifyZoomLevels",
                )
                CustomLayoutProperty(
                    "cesium:minimumZoomLevel",
                    build_fn=build_slider(
                        0, 30, type="int", constrain={"attr": "cesium:maximumZoomLevel", "type": "maximum"}
                    ),
                )
                CustomLayoutProperty(
                    "cesium:maximumZoomLevel",
                    build_fn=build_slider(
                        0, 30, type="int", constrain={"attr": "cesium:minimumZoomLevel", "type": "minimum"}
                    ),
                )
            build_common_raster_overlay_properties()

        return frame.apply(props)

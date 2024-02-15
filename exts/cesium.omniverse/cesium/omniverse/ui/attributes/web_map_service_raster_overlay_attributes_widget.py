import logging
from omni.kit.property.usd.custom_layout_helper import CustomLayoutFrame, CustomLayoutGroup, CustomLayoutProperty
from omni.kit.property.usd.usd_property_widget import SchemaPropertiesWidget
from cesium.usd.plugins.CesiumUsdSchemas import (
    WebMapServiceRasterOverlay as CesiumWebMapServiceRasterOverlay,
)
from .cesium_properties_widget_builder import build_slider, build_common_raster_overlay_properties


class CesiumWebMapServiceRasterOverlayAttributesWidget(SchemaPropertiesWidget):
    def __init__(self):
        super().__init__(
            "Cesium Web Map Service Raster Overlay Settings", CesiumWebMapServiceRasterOverlay, include_inherited=True
        )

        self._logger = logging.getLogger(__name__)

    def clean(self):
        super().clean()

    def _customize_props_layout(self, props):
        frame = CustomLayoutFrame(hide_extra=True)

        with frame:
            with CustomLayoutGroup("Base URL"):
                CustomLayoutProperty("cesium:baseUrl")
            with CustomLayoutGroup("Layers"):
                CustomLayoutProperty("cesium:layers")
            with CustomLayoutGroup("Tile Size"):
                CustomLayoutProperty("cesium:tileWidth", build_fn=build_slider(64, 2048, type="int"))
                CustomLayoutProperty("cesium:tileHeight", build_fn=build_slider(64, 2048, type="int"))
            with CustomLayoutGroup("Zoom Settings"):
                CustomLayoutProperty(
                    "cesium:minimumLevel",
                    build_fn=build_slider(
                        0, 30, type="int", constrain={"attr": "cesium:maximumLevel", "type": "maximum"}
                    ),
                )
                CustomLayoutProperty(
                    "cesium:maximumLevel",
                    build_fn=build_slider(
                        0, 30, type="int", constrain={"attr": "cesium:minimumLevel", "type": "minimum"}
                    ),
                )
            build_common_raster_overlay_properties()

        return frame.apply(props)

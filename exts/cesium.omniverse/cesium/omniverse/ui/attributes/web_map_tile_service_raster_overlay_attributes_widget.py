import logging
from omni.kit.property.usd.custom_layout_helper import CustomLayoutFrame, CustomLayoutGroup, CustomLayoutProperty
from omni.kit.property.usd.usd_property_widget import SchemaPropertiesWidget
from cesium.usd.plugins.CesiumUsdSchemas import (
    WebMapTileServiceRasterOverlay as CesiumWebMapTileServiceRasterOverlay,
)
from .cesium_properties_widget_builder import build_slider, build_common_raster_overlay_properties


class CesiumWebMapTileServiceRasterOverlayAttributesWidget(SchemaPropertiesWidget):
    def __init__(self):
        super().__init__(
            "Cesium Web Map Service Raster Overlay Settings",
            CesiumWebMapTileServiceRasterOverlay,
            include_inherited=True,
        )

        self._logger = logging.getLogger(__name__)

    def clean(self):
        super().clean()

    def _customize_props_layout(self, props):
        frame = CustomLayoutFrame(hide_extra=True)

        with frame:
            with CustomLayoutGroup("WMTS Settings"):
                CustomLayoutProperty("cesium:url")
                CustomLayoutProperty("cesium:layer")
                CustomLayoutProperty("cesium:style")
                CustomLayoutProperty("cesium:format")
                CustomLayoutProperty("cesium:tileMatrixSetId")
                CustomLayoutProperty("cesium:specifyTileMatrixSetLabels")
                # TODO: Tile Matrix Set Labels
                CustomLayoutProperty("cesium:tileMatrixSetLabelPrefix")
                CustomLayoutProperty("cesium:useWebMercatorProjection")
            with CustomLayoutGroup("Zoom Settings"):
                CustomLayoutProperty(
                    "cesium:minimumZoomLevel",
                    build_fn=build_slider(
                        0, 30, type="int", constrain={"attr": "cesium:maximumLevel", "type": "maximum"}
                    ),
                )
                CustomLayoutProperty(
                    "cesium:maximumZoomLevel",
                    build_fn=build_slider(
                        0, 30, type="int", constrain={"attr": "cesium:minimumLevel", "type": "minimum"}
                    ),
                )
            build_common_raster_overlay_properties()

        return frame.apply(props)

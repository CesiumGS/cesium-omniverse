import logging
import omni.kit.window.property
import omni.usd
import omni.ui
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
        self._stage = omni.usd.get_context().get_stage()

    def clean(self):
        super().clean()

    def _on_usd_changed(self, notice, stage):
        window = omni.kit.window.property.get_window()
        window.request_rebuild()

    def _customize_props_layout(self, props):
        frame = CustomLayoutFrame(hide_extra=True)

        prim_path = self._payload.get_paths()[0]
        webMapTileServiceRasterOverlay = CesiumWebMapTileServiceRasterOverlay.Get(self._stage, prim_path)
        specify_zoom_levels = webMapTileServiceRasterOverlay.GetSpecifyZoomLevelsAttr().Get()
        specify_tile_matrix_set_labels = webMapTileServiceRasterOverlay.GetSpecifyTileMatrixSetLabelsAttr().Get()
        specify_tiling_scheme = webMapTileServiceRasterOverlay.GetSpecifyTilingSchemeAttr().Get()

        with frame:
            with CustomLayoutGroup("WMTS Settings"):
                CustomLayoutProperty("cesium:url")
                CustomLayoutProperty("cesium:layer")
                CustomLayoutProperty("cesium:style")
                CustomLayoutProperty("cesium:format")
                with CustomLayoutGroup("Tiling Matrix Set Settings", collapsed=False):
                    CustomLayoutProperty("cesium:tileMatrixSetId")
                    CustomLayoutProperty("cesium:specifyTileMatrixSetLabels")
                    if specify_tile_matrix_set_labels:
                        CustomLayoutProperty("cesium:tileMatrixSetLabels")
                    else:
                        CustomLayoutProperty("cesium:tileMatrixSetLabelPrefix")
                    CustomLayoutProperty("cesium:useWebMercatorProjection")
                with CustomLayoutGroup("Tiling Scheme Settings", collapsed=False):
                    CustomLayoutProperty("cesium:specifyTilingScheme")
                    if specify_tiling_scheme:
                        CustomLayoutProperty("cesium:rootTilesX")
                        CustomLayoutProperty("cesium:rootTilesY")
                        CustomLayoutProperty("cesium:west")
                        CustomLayoutProperty("cesium:east")
                        CustomLayoutProperty("cesium:south")
                        CustomLayoutProperty("cesium:north")
                with CustomLayoutGroup("Zoom Settings", collapsed=False):
                    CustomLayoutProperty("cesium:specifyZoomLevels")
                    if specify_zoom_levels:
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

    def reset(self):
        if self._listener:
            self._listener.Revoke()
            self._listener = None
        super().reset()

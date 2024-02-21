import logging
import omni.usd
import omni.ui
import omni.kit.window.property
from omni.kit.property.usd.custom_layout_helper import CustomLayoutFrame, CustomLayoutGroup, CustomLayoutProperty
from omni.kit.property.usd.usd_property_widget import SchemaPropertiesWidget
from cesium.usd.plugins.CesiumUsdSchemas import (
    TileMapServiceRasterOverlay as CesiumTileMapServiceRasterOverlay,
)
from .cesium_properties_widget_builder import build_slider, build_common_raster_overlay_properties
from pxr import Usd, Tf


class CesiumTileMapServiceRasterOverlayAttributesWidget(SchemaPropertiesWidget):
    def __init__(self):
        super().__init__(
            "Cesium Tile Map Service Raster Overlay Settings",
            CesiumTileMapServiceRasterOverlay,
            include_inherited=True,
        )

        self._logger = logging.getLogger(__name__)

        self._listener = None
        self._props = None
        self._stage = omni.usd.get_context().get_stage()

    def clean(self):
        super().clean()

    def _on_usd_changed(self, notice, stage):
        window = omni.kit.window.property.get_window()
        window.request_rebuild()

    def _customize_props_layout(self, props):
        if not self._listener:
            self._listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self._on_usd_changed, self._stage)

        frame = CustomLayoutFrame(hide_extra=True)

        prim_path = self._payload.get_paths()[0]
        tileMapServiceRasterOverlay = CesiumTileMapServiceRasterOverlay.Get(self._stage, prim_path)
        specify_zoom_levels = tileMapServiceRasterOverlay.GetSpecifyZoomLevelsAttr().Get()

        with frame:
            with CustomLayoutGroup("URL"):
                CustomLayoutProperty("cesium:url")
            with CustomLayoutGroup("Zoom Settings"):
                CustomLayoutProperty(
                    "cesium:specifyZoomLevels",
                )
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

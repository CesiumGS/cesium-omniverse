import logging
from omni.kit.property.usd.custom_layout_helper import CustomLayoutFrame, CustomLayoutGroup, CustomLayoutProperty
from omni.kit.property.usd.usd_property_widget import SchemaPropertiesWidget
from cesium.usd.plugins.CesiumUsdSchemas import IonRasterOverlay as CesiumIonRasterOverlay


class CesiumIonRasterOverlayAttributesWidget(SchemaPropertiesWidget):
    def __init__(self):
        super().__init__("Cesium Ion Raster Overlay Settings", CesiumIonRasterOverlay, include_inherited=True)

        self._logger = logging.getLogger(__name__)

    def clean(self):
        super().clean()

    def _customize_props_layout(self, props):
        frame = CustomLayoutFrame(hide_extra=True)

        with frame:
            with CustomLayoutGroup("Credit Display"):
                CustomLayoutProperty("cesium:showCreditsOnScreen")
            with CustomLayoutGroup("Source"):
                CustomLayoutProperty("cesium:ionAssetId")
                CustomLayoutProperty("cesium:ionAccessToken")
                CustomLayoutProperty("cesium:ionServerBinding")
            with CustomLayoutGroup("Rendering"):
                CustomLayoutProperty("cesium:alpha")
                CustomLayoutProperty("cesium:overlayRenderMethod")

        return frame.apply(props)

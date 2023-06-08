import logging
from omni.kit.property.usd.custom_layout_helper import CustomLayoutFrame, CustomLayoutGroup, CustomLayoutProperty
from omni.kit.property.usd.usd_property_widget import SchemaPropertiesWidget
from cesium.usd.plugins.CesiumUsdSchemas import Georeference as CesiumGeoreference


class CesiumGeoreferenceSchemaAttributesWidget(SchemaPropertiesWidget):
    def __init__(self):
        super().__init__("Cesium Georeference", CesiumGeoreference, include_inherited=False)

        self._logger = logging.getLogger(__name__)

    def clean(self):
        super().clean()

    def _customize_props_layout(self, props):
        frame = CustomLayoutFrame(hide_extra=True)

        with frame:
            with CustomLayoutGroup("Georeference Origin Point Coordinates"):
                CustomLayoutProperty("cesium:georeferenceOrigin:latitude", "Latitude")
                CustomLayoutProperty("cesium:georeferenceOrigin:longitude", "Longitude")
                CustomLayoutProperty("cesium:georeferenceOrigin:height", "Height")

        return frame.apply(props)

import logging
from omni.kit.property.usd.custom_layout_helper import CustomLayoutFrame, CustomLayoutGroup, CustomLayoutProperty
from omni.kit.property.usd.usd_property_widget import SchemaPropertiesWidget
from cesium.usd.plugins.CesiumUsdSchemas import Data as CesiumData


class CesiumDataSchemaAttributesWidget(SchemaPropertiesWidget):
    def __init__(self):
        super().__init__("Cesium Settings", CesiumData, include_inherited=False)

        self._logger = logging.getLogger(__name__)

    def clean(self):
        super().clean()

    def _customize_props_layout(self, props):
        frame = CustomLayoutFrame(hide_extra=True)

        with frame:
            with CustomLayoutGroup("Access Tokens"):
                CustomLayoutProperty("cesium:projectDefaultIonAccessToken")
            with CustomLayoutGroup("Georeference Origin Point Coordinates"):
                CustomLayoutProperty("cesium:georeferenceOrigin:latitude", "Latitude")
                CustomLayoutProperty("cesium:georeferenceOrigin:longitude", "Longitude")
                CustomLayoutProperty("cesium:georeferenceOrigin:height", "Height")
            with CustomLayoutGroup("Debug Options", collapsed=True):
                CustomLayoutProperty("cesium:debug:disableMaterials")
                CustomLayoutProperty("cesium:debug:disableGeometryPool")
                CustomLayoutProperty("cesium:debug:disableMaterialPool")
                CustomLayoutProperty("cesium:debug:geometryPoolInitialCapacity")
                CustomLayoutProperty("cesium:debug:materialPoolInitialCapacity")

        return frame.apply(props)

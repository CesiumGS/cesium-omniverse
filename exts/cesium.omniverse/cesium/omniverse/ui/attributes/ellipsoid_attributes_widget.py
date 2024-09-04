import logging
from omni.kit.property.usd.custom_layout_helper import CustomLayoutFrame, CustomLayoutProperty
from omni.kit.property.usd.usd_property_widget import SchemaPropertiesWidget
from cesium.usd.plugins.CesiumUsdSchemas import Ellipsoid as CesiumEllipsoid


class CesiumEllipsoidAttributesWidget(SchemaPropertiesWidget):
    def __init__(self):
        super().__init__("Ellipsoid", CesiumEllipsoid, include_inherited=False)

        self._logger = logging.getLogger(__name__)

    def clean(self):
        super().clean()

    def _customize_props_layout(self, props):
        frame = CustomLayoutFrame(hide_extra=True)

        with frame:
            CustomLayoutProperty("cesium:radii")

        return frame.apply(props)

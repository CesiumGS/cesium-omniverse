import logging
from omni.kit.property.usd.custom_layout_helper import CustomLayoutFrame, CustomLayoutGroup, CustomLayoutProperty
from omni.kit.property.usd.usd_property_widget import SchemaPropertiesWidget
from cesium.usd.plugins.CesiumUsdSchemas import (
    Ellipsoid as CesiumEllipsoid,
    Georeference as CesiumGeoreference,
)


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
            with CustomLayoutGroup("Ellipsoid"):
                CustomLayoutProperty("cesium:ellipsoidBinding")

        return frame.apply(props)

    def _filter_props_to_build(self, props):
        filtered_props = super()._filter_props_to_build(props)
        filtered_props.extend(prop for prop in props if prop.GetName() == "cesium:ellipsoidBinding")
        return filtered_props

    def get_additional_kwargs(self, ui_attr):
        if ui_attr.prop_name == "cesium:ellipsoidBinding":
            return None, {"target_picker_filter_type_list": [CesiumEllipsoid], "targets_limit": 1}

        return None, {"targets_limit": 0}

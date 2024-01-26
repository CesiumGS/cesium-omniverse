import logging
from omni.kit.property.usd.custom_layout_helper import CustomLayoutFrame, CustomLayoutGroup, CustomLayoutProperty
from omni.kit.property.usd.usd_property_widget import SchemaPropertiesWidget
from ...bindings import ICesiumOmniverseInterface
from cesium.usd.plugins.CesiumUsdSchemas import (
    GlobeAnchorAPI as CesiumGlobeAnchorAPI,
    Georeference as CesiumGeoreference,
)


class CesiumGlobeAnchorAttributesWidget(SchemaPropertiesWidget):
    def __init__(self, _cesium_omniverse_interface: ICesiumOmniverseInterface):
        super().__init__("Cesium Globe Anchor", CesiumGlobeAnchorAPI, include_inherited=False)

        self._logger = logging.getLogger(__name__)

        self._cesium_omniverse_interface = _cesium_omniverse_interface

    def clean(self):
        super().clean()

    def _customize_props_layout(self, props):
        frame = CustomLayoutFrame(hide_extra=True)

        with frame:
            with CustomLayoutGroup("Options"):
                CustomLayoutProperty("cesium:anchor:adjustOrientationForGlobeWhenMoving")
                CustomLayoutProperty("cesium:anchor:detectTransformChanges")
                CustomLayoutProperty("cesium:anchor:georeferenceBinding")
            with CustomLayoutGroup("Global Positioning"):
                CustomLayoutProperty("cesium:anchor:latitude")
                CustomLayoutProperty("cesium:anchor:longitude")
                CustomLayoutProperty("cesium:anchor:height")
            with CustomLayoutGroup("Advanced Positioning", collapsed=True):
                CustomLayoutProperty("cesium:anchor:position")

        return frame.apply(props)

    def _filter_props_to_build(self, props):
        filtered_props = super()._filter_props_to_build(props)
        filtered_props.extend(prop for prop in props if prop.GetName() == "cesium:anchor:georeferenceBinding")
        return filtered_props

    def get_additional_kwargs(self, ui_attr):
        if ui_attr.prop_name == "cesium:anchor:georeferenceBinding":
            return None, {"target_picker_filter_type_list": [CesiumGeoreference], "targets_limit": 1}

        return None, {"targets_limit": 0}

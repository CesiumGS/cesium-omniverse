import logging
from omni.kit.property.usd.custom_layout_helper import CustomLayoutFrame, CustomLayoutGroup, CustomLayoutProperty
from omni.kit.property.usd.usd_property_widget import SchemaPropertiesWidget
from ...bindings import ICesiumOmniverseInterface
from cesium.usd.plugins.CesiumUsdSchemas import GlobalAnchorAPI as CesiumGlobalAnchorAPI


class CesiumGlobalAnchorAttributesWidget(SchemaPropertiesWidget):
    def __init__(self, _cesium_omniverse_interface: ICesiumOmniverseInterface):
        super().__init__("Cesium Global Anchor", CesiumGlobalAnchorAPI, include_inherited=False)

        self._logger = logging.getLogger(__name__)

        self._cesium_omniverse_interface = _cesium_omniverse_interface

    def clean(self):
        super().clean()

    def _customize_props_layout(self, props):
        # TODO: Lay this out better and give it some better functionality before merge.
        frame = CustomLayoutFrame(hide_extra=True)

        with frame:
            with CustomLayoutGroup("Options"):
                CustomLayoutProperty("cesium:anchor:adjustOrientationForGlobeWhenMoving")
                CustomLayoutProperty("cesium:anchor:detectTransformChanges")
            with CustomLayoutGroup("Global Positioning"):
                CustomLayoutProperty("cesium:anchor:longitude")
                CustomLayoutProperty("cesium:anchor:latitude")
                CustomLayoutProperty("cesium:anchor:height")
            with CustomLayoutGroup("Stage Positioning"):
                CustomLayoutProperty("cesium:anchor:position")
                CustomLayoutProperty("cesium:anchor:rotation")
                CustomLayoutProperty("cesium:anchor:scale")

        return frame.apply(props)

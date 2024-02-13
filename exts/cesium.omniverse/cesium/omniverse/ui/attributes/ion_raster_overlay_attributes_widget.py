import logging
from omni.kit.property.usd.custom_layout_helper import CustomLayoutFrame, CustomLayoutGroup, CustomLayoutProperty
from omni.kit.property.usd.usd_property_widget import SchemaPropertiesWidget
from cesium.usd.plugins.CesiumUsdSchemas import (
    IonRasterOverlay as CesiumIonRasterOverlay,
    IonServer as CesiumIonServer,
)
from .cesium_properties_widget_builder import build_common_raster_overlay_properties


class CesiumIonRasterOverlayAttributesWidget(SchemaPropertiesWidget):
    def __init__(self):
        super().__init__("Cesium Ion Raster Overlay Settings", CesiumIonRasterOverlay, include_inherited=True)

        self._logger = logging.getLogger(__name__)

    def clean(self):
        super().clean()

    def _customize_props_layout(self, props):
        frame = CustomLayoutFrame(hide_extra=True)

        with frame:
            with CustomLayoutGroup("Source"):
                CustomLayoutProperty("cesium:ionAssetId")
                CustomLayoutProperty("cesium:ionAccessToken")
                CustomLayoutProperty("cesium:ionServerBinding")
            build_common_raster_overlay_properties(add_overlay_render_method=True)

        return frame.apply(props)

    def _filter_props_to_build(self, props):
        filtered_props = super()._filter_props_to_build(props)
        filtered_props.extend(prop for prop in props if prop.GetName() == "cesium:ionServerBinding")
        return filtered_props

    def get_additional_kwargs(self, ui_attr):
        if ui_attr.prop_name == "cesium:ionServerBinding":
            return None, {"target_picker_filter_type_list": [CesiumIonServer], "targets_limit": 1}

        return None, {"targets_limit": 0}

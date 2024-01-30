import logging
import omni.ui as ui
from typing import List
from omni.kit.property.usd.custom_layout_helper import CustomLayoutFrame, CustomLayoutGroup, CustomLayoutProperty
from omni.kit.property.usd.usd_property_widget import SchemaPropertiesWidget
from cesium.usd.plugins.CesiumUsdSchemas import PolygonRasterOverlay as CesiumPolygonRasterOverlay
from pxr import Sdf, UsdGeom


class CesiumPolygonRasterOverlayAttributesWidget(SchemaPropertiesWidget):
    def __init__(self):
        super().__init__("Cesium Polygon Raster Overlay Settings", CesiumPolygonRasterOverlay, include_inherited=True)

        self._logger = logging.getLogger(__name__)

    def clean(self):
        super().clean()

    def _customize_props_layout(self, props):
        frame = CustomLayoutFrame(hide_extra=True)

        with frame:

            def update_range(stage, prim_paths):
                for path in prim_paths:
                    prim = stage.GetPrimAtPath(path)
                    attr = prim.GetAttribute("cesium:alpha") if prim else None
                    if prim and attr:
                        current_value = attr.Get()
                        new_value = max(0, min(current_value, 1.0))
                        attr.Set(new_value)

            def _build_slider(
                stage,
                attr_name,
                metadata,
                property_type,
                prim_paths: List[Sdf.Path],
                additional_label_kwargs={},
                additional_widget_kwargs={},
            ):
                from omni.kit.window.property.templates import HORIZONTAL_SPACING
                from omni.kit.property.usd.usd_attribute_model import UsdAttributeModel
                from omni.kit.property.usd.usd_property_widget import UsdPropertiesWidgetBuilder

                if not attr_name or not property_type:
                    return

                with ui.HStack(spacing=HORIZONTAL_SPACING):
                    model_kwargs = UsdPropertiesWidgetBuilder.get_attr_value_range_kwargs(metadata)
                    model = UsdAttributeModel(
                        stage,
                        [path.AppendProperty(attr_name) for path in prim_paths],
                        False,
                        metadata,
                        **model_kwargs,
                    )
                    UsdPropertiesWidgetBuilder.create_label(attr_name, metadata, additional_label_kwargs)
                    widget_kwargs = {"model": model, "min": 0.0, "max": 1.0}
                    widget_kwargs.update(UsdPropertiesWidgetBuilder.get_attr_value_soft_range_kwargs(metadata))

                    if additional_widget_kwargs:
                        widget_kwargs.update(additional_widget_kwargs)
                    with ui.ZStack():
                        value_widget = UsdPropertiesWidgetBuilder.create_drag_or_slider(
                            ui.FloatDrag, ui.FloatSlider, **widget_kwargs
                        )
                        mixed_overlay = UsdPropertiesWidgetBuilder.create_mixed_text_overlay()
                    UsdPropertiesWidgetBuilder.create_control_state(
                        value_widget=value_widget, mixed_overlay=mixed_overlay, **widget_kwargs
                    )
                    model.add_value_changed_fn(lambda m, s=stage, p=prim_paths: update_range(s, p))

                    return model

            with CustomLayoutGroup("Credit Display"):
                CustomLayoutProperty("cesium:showCreditsOnScreen")
            with CustomLayoutGroup("Rendering"):
                CustomLayoutProperty("cesium:alpha", build_fn=_build_slider)
                CustomLayoutProperty("cesium:overlayRenderMethod")
            with CustomLayoutGroup("Invert Selection"):
                CustomLayoutProperty("cesium:invertSelection")
            with CustomLayoutGroup("Cartographic Polygons"):
                CustomLayoutProperty("cesium:cartographicPolygonBinding")

        return frame.apply(props)

    def _filter_props_to_build(self, props):
        filtered_props = super()._filter_props_to_build(props)
        filtered_props.extend(
            prop
            for prop in props
            if prop.GetName() == "cesium:cartographicPolygonBinding"
        )
        return filtered_props

    def get_additional_kwargs(self, ui_attr):
        if ui_attr.prop_name == "cesium:cartographicPolygonBinding":
            return None, {"target_picker_filter_type_list": [UsdGeom.BasisCurves]}

        return None, {"targets_limit": 0}

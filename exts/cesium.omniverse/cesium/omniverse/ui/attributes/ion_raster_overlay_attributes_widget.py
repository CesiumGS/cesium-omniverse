import logging
import omni.ui as ui
from typing import List
from omni.kit.property.usd.custom_layout_helper import CustomLayoutFrame, CustomLayoutGroup, CustomLayoutProperty
from omni.kit.property.usd.usd_property_widget import SchemaPropertiesWidget
from cesium.usd.plugins.CesiumUsdSchemas import IonRasterOverlay as CesiumIonRasterOverlay
from pxr import Sdf


class CesiumIonRasterOverlayAttributesWidget(SchemaPropertiesWidget):
    def __init__(self):
        super().__init__("Cesium Ion Raster Overlay Settings", CesiumIonRasterOverlay, include_inherited=True)

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
            with CustomLayoutGroup("Source"):
                CustomLayoutProperty("cesium:ionAssetId")
                CustomLayoutProperty("cesium:ionAccessToken")
                CustomLayoutProperty("cesium:ionServerBinding")
            with CustomLayoutGroup("Rendering"):
                CustomLayoutProperty("cesium:alpha", build_fn=_build_slider)
                CustomLayoutProperty("cesium:overlayRenderMethod")

        return frame.apply(props)

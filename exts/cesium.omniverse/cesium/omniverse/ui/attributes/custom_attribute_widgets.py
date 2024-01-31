from typing import List
import omni.ui as ui
from pxr import Sdf


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
        widget_kwargs = {"model": model}
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

        return model


def build_slider(min_value, max_value):
    def custom_slider(stage, attr_name, metadata, property_type, prim_paths, *args, **kwargs):
        additional_widget_kwargs = {"min": min_value, "max": max_value}
        additional_widget_kwargs.update(kwargs)
        return _build_slider(
            stage, attr_name, metadata, property_type, prim_paths, additional_widget_kwargs=additional_widget_kwargs
        )

    return custom_slider

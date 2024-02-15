from typing import List
import omni.ui as ui
from pxr import Sdf
from functools import partial
from omni.kit.property.usd.custom_layout_helper import CustomLayoutGroup, CustomLayoutProperty


def update_range(stage, prim_paths, constrain, attr_name):
    min_val = max_val = None

    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        attr = prim.GetAttribute(constrain["attr"]) if prim else None
        if prim and attr:
            if constrain["type"] == "minimum":
                min_val = attr.Get()
            else:
                max_val = attr.Get()
            break

    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        attr = prim.GetAttribute(attr_name) if prim else None
        if prim and attr:
            val = attr.Get()
            if min_val:
                val = max(min_val, val)
            elif max_val:
                val = min(max_val, val)
            attr.Set(val)
            break


def _build_slider(
    stage,
    attr_name,
    metadata,
    property_type,
    prim_paths: List[Sdf.Path],
    type="float",
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
            if type == "float":
                value_widget = UsdPropertiesWidgetBuilder.create_drag_or_slider(
                    ui.FloatDrag, ui.FloatSlider, **widget_kwargs
                )
            else:
                value_widget = UsdPropertiesWidgetBuilder.create_drag_or_slider(
                    ui.IntDrag, ui.IntSlider, **widget_kwargs
                )
            mixed_overlay = UsdPropertiesWidgetBuilder.create_mixed_text_overlay()
        UsdPropertiesWidgetBuilder.create_control_state(
            value_widget=value_widget, mixed_overlay=mixed_overlay, **widget_kwargs
        )

        if len(additional_widget_kwargs["constrain"]) == 2:
            callback = partial(update_range, stage, prim_paths, additional_widget_kwargs["constrain"], attr_name)
            model.add_value_changed_fn(lambda m: callback())
        return model


def build_slider(min_value, max_value, type="float", constrain={}):
    if type not in ["int", "float"]:
        raise ValueError("'type' must be 'int' or 'float'")

    if len(constrain) not in [0, 2]:
        raise ValueError("'constrain' must be empty or a {'attr': ___, 'type': ___} dictionary")
        if constrain[1] not in ["minimum", "maximum"]:
            raise ValueError("constrain['type'] must be 'minimum' or 'maximum'")

    def custom_slider(stage, attr_name, metadata, property_type, prim_paths, *args, **kwargs):
        additional_widget_kwargs = {"min": min_value, "max": max_value, "constrain": constrain}
        additional_widget_kwargs.update(kwargs)
        return _build_slider(
            stage,
            attr_name,
            metadata,
            property_type,
            prim_paths,
            additional_widget_kwargs=additional_widget_kwargs,
            type=type,
        )

    return custom_slider


def build_common_raster_overlay_properties(add_overlay_render_method=False):
    with CustomLayoutGroup("Rendering"):
        CustomLayoutProperty("cesium:alpha", build_fn=build_slider(0, 1))
        if add_overlay_render_method:
            CustomLayoutProperty("cesium:overlayRenderMethod")
    CustomLayoutProperty("cesium:maximumScreenSpaceError")
    CustomLayoutProperty("cesium:maximumTextureSize")
    CustomLayoutProperty("cesium:maximumSimultaneousTileLoads")
    CustomLayoutProperty("cesium:subTileCacheBytes")
    CustomLayoutProperty("cesium:showCreditsOnScreen")

import omni.kit.commands
import omni.kit.context_menu
from pxr import UsdGeom
from .utils import (
    convert_curves_to_polygons,
)
from cesium.usd.plugins.CesiumUsdSchemas import CartographicPolygon


class ContextMenu:
    @classmethod
    def startup(cls):
        # Adds the option to the viewport context menu
        cls._register_viewport_context_menu()

        # Adds the option to the stage window context menu
        manager = omni.kit.app.get_app().get_extension_manager()
        cls._hooks = manager.subscribe_to_extension_enable(
            on_enable_fn=lambda _: cls._register_stage_context_menu(),
            on_disable_fn=lambda _: cls._unregister_stage_context_menu(),
            ext_name="omni.kit.window.stage",
            hook_name="listener",
        )

    @classmethod
    def shutdown(cls):
        cls._unregister_viewport_context_menu()
        cls._unregister_stage_context_menu()

    @classmethod
    def _add_convert_curve_menu_impl(cls, extension: str):
        menu = {
            "name": "Convert to Cesium Cartographic Polygon",
            "glyph": "copy.svg",
            "show_fn": cls._is_selected_all_curve,
            "onclick_fn": cls._convert_curves,
        }
        return omni.kit.context_menu.add_menu(menu, "MENU", extension)

    @classmethod
    def _register_viewport_context_menu(cls):
        cls._convert_curve_viewport_entry = cls._add_convert_curve_menu_impl("omni.kit.window.viewport")

    @classmethod
    def _unregister_viewport_context_menu(cls):
        cls._convert_curve_viewport_entry = None

    @classmethod
    def _register_stage_context_menu(cls):
        cls._convert_curve_stage_entry = cls._add_convert_curve_menu_impl("omni.kit.widget.stage")

    @classmethod
    def _unregister_stage_context_menu(cls):
        cls._convert_curve_stage_entry = None

    @staticmethod
    def _is_selected_all_curve(object: dict):
        prim_list = object.get("prim_list", [])
        if not prim_list:
            return False

        for prim in prim_list:
            if not prim.IsA(UsdGeom.BasisCurves):
                return False
            if prim.IsA(CartographicPolygon):
                return False

        return True

    @classmethod
    def _convert_curves(cls, object: dict):
        prim_list = object.get("prim_list", [])
        prim_list_str = [prim.GetPath().pathString for prim in prim_list if prim.IsA(UsdGeom.BasisCurves)]
        convert_curves_to_polygons(prim_list_str)

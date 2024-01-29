import omni.kit.commands
import omni.kit.context_menu
from pxr import UsdGeom
from .clipping import CesiumCartographicPolygonUtility
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
            "name": "Create Cesium Cartographic Polygon from BasisCurve",
            "glyph": "copy.svg",
            "show_fn": cls._is_selected_all_curve,
            "onclick_fn": cls._convert_curves,
        }
        return omni.kit.context_menu.add_menu(menu, "MENU", extension)

    @classmethod
    def _add_create_footprint_menu_impl(cls, extension: str):
        menu = {
            "name": "Create BasisCurves from Prim footprint",
            "glyph": "copy.svg",
            "show_fn": cls._selected_have_mesh,  # Show for all prims
            "onclick_fn": cls._create_footprints,
        }
        return omni.kit.context_menu.add_menu(menu, "MENU", extension)

    @classmethod
    def _register_viewport_context_menu(cls):
        cls._convert_curve_viewport_entry = cls._add_convert_curve_menu_impl("omni.kit.window.viewport")
        cls._create_footprint_viewport_entry = cls._add_create_footprint_menu_impl("omni.kit.window.viewport")

    @classmethod
    def _unregister_viewport_context_menu(cls):
        cls._convert_curve_viewport_entry = None
        cls._create_footprint_viewport_entry = None

    @classmethod
    def _register_stage_context_menu(cls):
        cls._convert_curve_stage_entry = cls._add_convert_curve_menu_impl("omni.kit.widget.stage")
        cls._create_footprint_stage_entry = cls._add_create_footprint_menu_impl("omni.kit.widget.stage")

    @classmethod
    def _unregister_stage_context_menu(cls):
        cls._convert_curve_stage_entry = None
        cls._create_footprint_stage_entry = None

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

    @staticmethod
    def find_mesh_in_children(parent_prim):
        if parent_prim.IsA(UsdGeom.Mesh):
            return True

        for child_prim in parent_prim.GetAllChildren():
            if ContextMenu.find_mesh_in_children(child_prim):
                return True

        return False

    @staticmethod
    def _selected_have_mesh(object: dict):
        prim_list = object.get("prim_list", [])
        if not prim_list:
            return False

        for prim in prim_list:
            if not ContextMenu.find_mesh_in_children(prim):
                return False

        return True

    @classmethod
    def _convert_curves(cls, object: dict):
        prim_list = object.get("prim_list", [])
        prim_list_str = [prim.GetPath().pathString for prim in prim_list if prim.IsA(UsdGeom.BasisCurves)]
        CesiumCartographicPolygonUtility.create_cartographic_polygons_from_curves(prim_list_str)

    @classmethod
    def _create_footprints(cls, object: dict):
        prim_list = object.get("prim_list", [])
        prim_list_str = [prim.GetPath().pathString for prim in prim_list]
        # create_prim_footprints(prim_list_str)
        CesiumCartographicPolygonUtility.create_prim_footprints(prim_list_str)

from functools import partial
import logging
import omni.kit.context_menu
from omni.kit.property.usd import PrimPathWidget, PrimSelectionPayload
from omni.kit.window.property import get_window as get_property_window
from pxr import Tf, UsdGeom
from cesium.usd.plugins.CesiumUsdSchemas import Tileset as CesiumTileset
from ..api.globe_anchor import anchor_xform_at_path
from ..bindings import ICesiumOmniverseInterface


class CesiumAddMenuController:
    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface):
        self._logger = logging.getLogger(__name__)
        self._cesium_omniverse_interface = cesium_omniverse_interface

        context_menu = omni.kit.context_menu.get_instance()
        if context_menu is None:
            self._logger.error("Cannot add Cesium options to Add menu when context_menu is disabled.")
            return

        self._items_added = [
            PrimPathWidget.add_button_menu_entry(
                "Cesium/Globe Anchor",
                show_fn=partial(self._show_add_globe_anchor, context_menu=context_menu, usd_type=UsdGeom.Xformable),
                onclick_fn=self._add_globe_anchor_api,
            )
        ]

    def destroy(self):
        for item in self._items_added:
            PrimPathWidget.remove_button_menu_entry(item)
        self._items_added.clear()

    def _add_globe_anchor_api(self, payload: PrimSelectionPayload):
        for path in payload:
            anchor_xform_at_path(path)
            get_property_window().request_rebuild()

    @staticmethod
    def _show_add_globe_anchor(objects: dict, context_menu: omni.kit.context_menu, usd_type: Tf.Type) -> bool:
        return context_menu.prim_is_type(objects, type=usd_type) and not context_menu.prim_is_type(
            objects, type=CesiumTileset
        )

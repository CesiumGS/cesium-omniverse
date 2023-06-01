from functools import partial
import logging
import omni.kit.context_menu
from omni.kit.property.usd import PrimPathWidget, PrimSelectionPayload
from pxr import UsdGeom
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
                "Cesium/Global Anchor",
                show_fn=partial(context_menu.prim_is_type, type=UsdGeom.Xformable),
                onclick_fn=self._add_global_anchor_api,
            )
        ]

    def destroy(self):
        for item in self._items_added:
            PrimPathWidget.remove_button_menu_entry(item)
        self._items_added.clear()

    def _add_global_anchor_api(self, payload: PrimSelectionPayload):
        # TODO: Call the C++ layer to add the the API with the current position as global.
        import pprint

        self._logger.warning(pprint.pformat(payload.get_paths()))

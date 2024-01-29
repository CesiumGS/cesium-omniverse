from functools import partial
import logging
import omni.kit.context_menu
from omni.kit.property.usd import PrimPathWidget, PrimSelectionPayload
from omni.kit.window.property import get_window as get_property_window
import omni.usd
from pxr import Sdf, Tf, UsdGeom
from cesium.usd.plugins.CesiumUsdSchemas import (
    Tileset as CesiumTileset,
    PolygonRasterOverlay as CesiumPolygonRasterOverlay,
    IonRasterOverlay as CesiumIonRasterOverlay,
)
from ..usdUtils import add_globe_anchor_to_prim
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
            ),
            PrimPathWidget.add_button_menu_entry(
                "Cesium/Ion Raster Overlay",
                show_fn=partial(self._show_add_raster_overlay, context_menu=context_menu, usd_type=CesiumTileset),
                onclick_fn=self._add_ion_raster_overlay,
            ),
            PrimPathWidget.add_button_menu_entry(
                "Cesium/Polygon Raster Overlay",
                show_fn=partial(self._show_add_raster_overlay, context_menu=context_menu, usd_type=CesiumTileset),
                onclick_fn=self._add_polygon_raster_overlay,
            ),
        ]

    def destroy(self):
        for item in self._items_added:
            PrimPathWidget.remove_button_menu_entry(item)
        self._items_added.clear()

    def _add_globe_anchor_api(self, payload: PrimSelectionPayload):
        for path in payload:
            add_globe_anchor_to_prim(path)
            get_property_window().request_rebuild()

    def _add_ion_raster_overlay(self, payload: PrimSelectionPayload):
        stage = omni.usd.get_context().get_stage()
        for path in payload:
            child_path = Sdf.Path(path).AppendPath("ion_raster_overlay")
            ion_raster_overlay_path: str = omni.usd.get_stage_next_free_path(stage, child_path, False)
            CesiumIonRasterOverlay.Define(stage, ion_raster_overlay_path)
            get_property_window().request_rebuild()

    def _add_polygon_raster_overlay(self, payload: PrimSelectionPayload):
        stage = omni.usd.get_context().get_stage()
        for path in payload:
            child_path = Sdf.Path(path).AppendPath("polygon_raster_overlay")
            polygon_raster_overlay_path: str = omni.usd.get_stage_next_free_path(stage, child_path, False)
            CesiumPolygonRasterOverlay.Define(stage, polygon_raster_overlay_path)
            get_property_window().request_rebuild()

    @staticmethod
    def _show_add_globe_anchor(objects: dict, context_menu: omni.kit.context_menu, usd_type: Tf.Type) -> bool:
        return context_menu.prim_is_type(objects, type=usd_type) and not context_menu.prim_is_type(
            objects, type=CesiumTileset
        )

    @staticmethod
    def _show_add_raster_overlay(objects: dict, context_menu: omni.kit.context_menu, usd_type: Tf.Type) -> bool:
        return context_menu.prim_is_type(objects, type=usd_type)

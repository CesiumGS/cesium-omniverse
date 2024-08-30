import logging
import omni.kit.window.property
from .attributes import (
    CesiumDataSchemaAttributesWidget,
    CesiumEllipsoidAttributesWidget,
    CesiumGeoreferenceSchemaAttributesWidget,
    CesiumTilesetAttributesWidget,
    CesiumGlobeAnchorAttributesWidget,
    CesiumIonServerAttributesWidget,
    CesiumIonRasterOverlayAttributesWidget,
    CesiumPolygonRasterOverlayAttributesWidget,
    CesiumTileMapServiceRasterOverlayAttributesWidget,
    CesiumWebMapServiceRasterOverlayAttributesWidget,
    CesiumWebMapTileServiceRasterOverlayAttributesWidget,
)
from ..bindings import ICesiumOmniverseInterface


class CesiumAttributesWidgetController:
    """
    This is designed as a helpful function for separating out the registration and
    unregistration of Cesium's attributes widgets.
    """

    def __init__(self, _cesium_omniverse_interface: ICesiumOmniverseInterface):
        self._cesium_omniverse_interface = _cesium_omniverse_interface
        self._logger = logging.getLogger(__name__)

        self._register_data_attributes_widget()
        self._register_ellipsoid_attributes_widget()
        self._register_georeference_attributes_widget()
        self._register_tileset_attributes_widget()
        self._register_global_anchor_attributes_widget()
        self._register_ion_server_attributes_widget()
        self._register_ion_raster_overlay_attributes_widget()
        self._register_polygon_raster_overlay_attributes_widget()
        self._register_tile_map_service_raster_overlay_attributes_widget()
        self._register_web_map_service_raster_overlay_attributes_widget()
        self._register_web_map_tile_service_raster_overlay_attributes_widget()

    def destroy(self):
        self._unregister_data_attributes_widget()
        self._unregister_ellipsoid_attributes_widget()
        self._unregister_georeference_attributes_widget()
        self._unregister_tileset_attributes_widget()
        self._unregister_global_anchor_attributes_widget()
        self._unregister_ion_server_attributes_widget()
        self._unregister_ion_raster_overlay_attributes_widget()
        self._unregister_polygon_raster_overlay_attributes_widget()
        self._unregister_tile_map_service_raster_overlay_attributes_widget()
        self._unregister_web_map_service_raster_overlay_attributes_widget()
        self._unregister_web_map_tile_service_raster_overlay_attributes_widget()

    @staticmethod
    def _register_data_attributes_widget():
        window = omni.kit.window.property.get_window()
        if window is not None:
            window.register_widget("prim", "cesiumData", CesiumDataSchemaAttributesWidget())

    @staticmethod
    def _unregister_data_attributes_widget():
        window = omni.kit.window.property.get_window()
        if window is not None:
            window.unregister_widget("prim", "cesiumData")

    @staticmethod
    def _register_ellipsoid_attributes_widget():
        window = omni.kit.window.property.get_window()
        if window is not None:
            window.register_widget("prim", "cesiumEllipsoid", CesiumEllipsoidAttributesWidget())

    @staticmethod
    def _unregister_ellipsoid_attributes_widget():
        window = omni.kit.window.property.get_window()
        if window is not None:
            window.unregister_widget("prim", "cesiumEllipsoid")

    @staticmethod
    def _register_georeference_attributes_widget():
        window = omni.kit.window.property.get_window()
        if window is not None:
            window.register_widget("prim", "cesiumGeoreference", CesiumGeoreferenceSchemaAttributesWidget())

    @staticmethod
    def _unregister_georeference_attributes_widget():
        window = omni.kit.window.property.get_window()
        if window is not None:
            window.unregister_widget("prim", "cesiumGeoreference")

    def _register_tileset_attributes_widget(self):
        window = omni.kit.window.property.get_window()
        if window is not None:
            window.register_widget(
                "prim", "cesiumTileset", CesiumTilesetAttributesWidget(self._cesium_omniverse_interface)
            )

    @staticmethod
    def _unregister_tileset_attributes_widget():
        window = omni.kit.window.property.get_window()
        if window is not None:
            window.unregister_widget("prim", "cesiumTileset")

    @staticmethod
    def _register_ion_raster_overlay_attributes_widget():
        window = omni.kit.window.property.get_window()
        if window is not None:
            window.register_widget("prim", "cesiumIonRasterOverlay", CesiumIonRasterOverlayAttributesWidget())

    @staticmethod
    def _unregister_ion_raster_overlay_attributes_widget():
        window = omni.kit.window.property.get_window()
        if window is not None:
            window.unregister_widget("prim", "cesiumIonRasterOverlay")

    @staticmethod
    def _register_polygon_raster_overlay_attributes_widget():
        window = omni.kit.window.property.get_window()
        if window is not None:
            window.register_widget("prim", "cesiumPolygonRasterOverlay", CesiumPolygonRasterOverlayAttributesWidget())

    @staticmethod
    def _unregister_polygon_raster_overlay_attributes_widget():
        window = omni.kit.window.property.get_window()
        if window is not None:
            window.unregister_widget("prim", "cesiumPolygonRasterOverlay")

    @staticmethod
    def _register_web_map_service_raster_overlay_attributes_widget():
        window = omni.kit.window.property.get_window()
        if window is not None:
            window.register_widget(
                "prim", "cesiumWebMapServiceRasterOverlay", CesiumWebMapServiceRasterOverlayAttributesWidget()
            )

    @staticmethod
    def _register_tile_map_service_raster_overlay_attributes_widget():
        window = omni.kit.window.property.get_window()
        if window is not None:
            window.register_widget(
                "prim", "cesiumTileMapServiceRasterOverlay", CesiumTileMapServiceRasterOverlayAttributesWidget()
            )

    @staticmethod
    def _register_web_map_tile_service_raster_overlay_attributes_widget():
        window = omni.kit.window.property.get_window()
        if window is not None:
            window.register_widget(
                "prim", "cesiumWebMapTileServiceRasterOverlay", CesiumWebMapTileServiceRasterOverlayAttributesWidget()
            )

    @staticmethod
    def _unregister_web_map_service_raster_overlay_attributes_widget():
        window = omni.kit.window.property.get_window()
        if window is not None:
            window.unregister_widget("prim", "cesiumWebMapServiceRasterOverlay")

    @staticmethod
    def _unregister_tile_map_service_raster_overlay_attributes_widget():
        window = omni.kit.window.property.get_window()
        if window is not None:
            window.unregister_widget("prim", "cesiumTileMapServiceRasterOverlay")

    @staticmethod
    def _unregister_web_map_tile_service_raster_overlay_attributes_widget():
        window = omni.kit.window.property.get_window()
        if window is not None:
            window.unregister_widget("prim", "cesiumWebMapTileServiceRasterOverlay")

    def _register_global_anchor_attributes_widget(self):
        window = omni.kit.window.property.get_window()
        if window is not None:
            window.register_widget(
                "prim", "cesiumGlobeAnchorAPI", CesiumGlobeAnchorAttributesWidget(self._cesium_omniverse_interface)
            )

    @staticmethod
    def _unregister_global_anchor_attributes_widget():
        window = omni.kit.window.property.get_window()
        if window is not None:
            window.unregister_widget("prim", "cesiumGlobalAnchorAPI")

    def _register_ion_server_attributes_widget(self):
        window = omni.kit.window.property.get_window()
        if window is not None:
            window.register_widget(
                "prim", "cesiumIonServer", CesiumIonServerAttributesWidget(self._cesium_omniverse_interface)
            )

    @staticmethod
    def _unregister_ion_server_attributes_widget():
        window = omni.kit.window.property.get_window()
        if window is not None:
            window.unregister_widget("prim", "cesiumIonServer")

from enum import Enum
import logging
import omni.ui as ui
from typing import Optional
from ..bindings import ICesiumOmniverseInterface
from .troubleshooter_window import CesiumTroubleshooterWindow


class Tileset(Enum):
    """Possible tilesets for use with Cesium for Omniverse."""

    CESIUM_WORLD_TERRAIN = 0
    BING_MAPS = 1
    CAPE_CANAVERAL = 2


class CesiumOmniverseDebugWindow(ui.Window):
    WINDOW_NAME = "Cesium Debugging"
    MENU_PATH = f"Window/Cesium/{WINDOW_NAME}"

    _logger: logging.Logger
    _cesium_omniverse_interface: Optional[ICesiumOmniverseInterface] = None

    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, title: str, **kwargs):
        super().__init__(title, **kwargs)

        self._logger = logging.getLogger(__name__)
        self._cesium_omniverse_interface = cesium_omniverse_interface
        self._cesium_message_field: Optional[ui.SimpleStringModel] = None

        # Set the function that is called to build widgets when the window is visible
        self.frame.set_build_fn(self._build_fn)

    def destroy(self):
        # It will destroy all the children
        super().destroy()

    def _build_fn(self):
        """Builds out the UI buttons and their handlers."""

        def add_maxar_3d_surface_model():
            """Adds the Maxar data of Cape Canaveral to the stage."""

            # Cape Canaveral
            self._cesium_omniverse_interface.set_georeference_origin(-80.53, 28.46, -30.0)

            self._cesium_omniverse_interface.add_tileset_ion(
                "Cape Canaveral",
                1387142,
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyMjRhYzI0Yi1kNWEwLTQ4ZWYtYjdmZC1hY2JmYWIzYmFiMGUiLCJpZCI6NDQsImlhdCI6MTY2NzQ4OTg0N30.du0tvWptgLWsvM1Gnbv3Zw_pDAOILg1Wr6s2sgK-qlM",
            )

        def add_cesium_world_terrain():
            """Adds the standard Cesium World Terrain to the stage."""

            # Cesium HQ
            self._cesium_omniverse_interface.set_georeference_origin(-75.1564977, 39.9501464, 150.0)

            tileset_id = self._cesium_omniverse_interface.add_tileset_ion(
                "Cesium World Terrain",
                1,
            )

            self._cesium_omniverse_interface.add_ion_raster_overlay(
                tileset_id,
                "Layer",
                3954,
            )

        def add_bing_maps_terrain():
            """Adds the Bing Maps & Cesium Terrain to the stage."""

            # Cesium HQ
            self._cesium_omniverse_interface.set_georeference_origin(-75.1564977, 39.9501464, 150.0)

            tileset_id = self._cesium_omniverse_interface.add_tileset_ion(
                "Bing Maps",
                1,
            )

            self._cesium_omniverse_interface.add_ion_raster_overlay(
                tileset_id,
                "Layer",
                2,
            )

        def create_tileset(tileset=Tileset.CESIUM_WORLD_TERRAIN):
            """Creates the desired tileset on the stage.

            Parameters:
                tileset (Tileset): The desired tileset specified using the Tileset enumeration.
            """

            if tileset is Tileset.CAPE_CANAVERAL:
                add_maxar_3d_surface_model()
            elif tileset is Tileset.BING_MAPS:
                add_bing_maps_terrain()
            else:  # Terrain is Cesium World Terrain
                add_cesium_world_terrain()

        def remove_all_tilesets():
            """Removes all tilesets from the stage."""

            tilesets = self._cesium_omniverse_interface.get_all_tileset_ids_and_paths()

            for tileset_id, _ in tilesets:
                self._cesium_omniverse_interface.remove_tileset(tileset_id)

        def reload_all_tilesets():
            """Reloads all tilesets."""

            tilesets = self._cesium_omniverse_interface.get_all_tileset_ids_and_paths()

            for tileset_id, _ in tilesets:
                self._cesium_omniverse_interface.reload_tileset(tileset_id)

        def print_fabric_stage():
            """Prints the contents of the Fabric stage to a text field."""

            fabric_stage = self._cesium_omniverse_interface.print_fabric_stage()
            self._cesium_message_field.set_value(fabric_stage)

        def open_troubleshooting_window():
            CesiumTroubleshooterWindow(self._cesium_omniverse_interface, "Testing", 1, 0, "Testing")

        with ui.VStack():
            ui.Button(
                "Create Cesium World Terrain Tileset", clicked_fn=lambda: create_tileset(Tileset.CESIUM_WORLD_TERRAIN)
            )
            ui.Button("Create Bing Maps Tileset", clicked_fn=lambda: create_tileset(Tileset.BING_MAPS))
            ui.Button("Create Cape Canaveral Tileset", clicked_fn=lambda: create_tileset(Tileset.CAPE_CANAVERAL))
            ui.Button("Remove all Tilesets", clicked_fn=lambda: remove_all_tilesets())
            ui.Button("Reload all Tilesets", clicked_fn=lambda: reload_all_tilesets())
            ui.Button("Open Troubleshooter", clicked_fn=lambda: open_troubleshooting_window())
            ui.Button("Print Fabric stage", clicked_fn=lambda: print_fabric_stage())
            with ui.VStack():
                self._cesium_message_field = ui.SimpleStringModel("")
                ui.StringField(self._cesium_message_field, multiline=True, read_only=True)

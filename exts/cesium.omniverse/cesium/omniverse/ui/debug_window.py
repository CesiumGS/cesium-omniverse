from ..bindings.CesiumOmniversePythonBindings import *
from carb.events._events import ISubscription
import carb.settings as omni_settings
from enum import Enum
import logging
import omni.ui as ui
import omni.usd
from omni.kit.viewport.utility import get_active_viewport, get_active_viewport_camera_path
import omni.kit.app as omni_app
import omni.kit.commands as omni_commands
from pxr import Gf, Sdf


class Tileset(Enum):
    """Possible tilesets for use with Cesium for Omniverse."""

    CESIUM_WORLD_TERRAIN = 0
    BING_MAPS = 1
    CAPE_CANAVERAL = 2


class CesiumOmniverseDebugWindow(ui.Window):

    WINDOW_NAME = "Cesium Debugging"
    MENU_PATH = f"Window/Cesium/{WINDOW_NAME}"

    _subscription_handle: ISubscription = None
    _logger: logging.Logger
    _cesium_omniverse_interface: ICesiumOmniverseInterface = None

    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, title: str, **kwargs):
        super().__init__(title, **kwargs)

        self._logger = logging.getLogger(__name__)
        self._cesium_omniverse_interface = cesium_omniverse_interface

        # Set the function that is called to build widgets when the window is visible
        self.frame.set_build_fn(self._build_fn)

        self._tilesets = []

    def destroy_subscription(self):
        """Unsubscribes from the subscription handler for frame updates."""

        if self._subscription_handle is not None:
            self._subscription_handle.unsubscribe()
            self._subscription_handle = None

    def destroy(self):
        # It will destroy all the children
        self.destroy_subscription()
        super().destroy()

    def update_far_plane(self):
        """Sets the Far Plane to a very high number."""

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        camera_prim = stage.GetPrimAtPath(get_active_viewport_camera_path())
        omni_commands.execute(
            "ChangeProperty",
            prop_path=Sdf.Path("/OmniverseKit_Persp.clippingRange"),
            value=Gf.Vec2f(1.0, 10000000000.0),
            prev=camera_prim.GetAttribute("clippingRange").Get(),
        )

    def _build_fn(self):
        """Builds out the UI buttons and their handlers."""

        def on_update_frame(e):
            """Actions performed on each frame update."""

            viewport = get_active_viewport()
            for tileset in self._tilesets:
                self._cesium_omniverse_interface.updateFrame(
                    tileset,
                    viewport.view,
                    viewport.projection,
                    float(viewport.resolution[0]),
                    float(viewport.resolution[1]),
                )

        def start_update_frame():
            """Starts updating the frame, resulting in tileset updates."""

            app = omni_app.get_app()
            omni_settings.get_settings().set("/rtx/hydra/TBNFrameMode", 1)
            # Disabling Texture Streaming is a workaround for issues with Kit 104.1. We should remove this as soon as
            #   the issue is fixed.
            omni_settings.get_settings().set("/rtx-transient/resourcemanager/enableTextureStreaming", False)
            self._subscription_handle = app.get_update_event_stream().create_subscription_to_pop(
                on_update_frame, name="cesium_update_frame"
            )

        def stop_update_frame():
            """Stops updating the frame, thereby stopping tileset updates."""

            self.destroy_subscription()

        def add_maxar_3d_surface_model():
            """Adds the Maxar data of Cape Canaveral to the stage."""

            # Cape Canaveral
            self._cesium_omniverse_interface.setGeoreferenceOrigin(-80.53, 28.46, -30.0)

            stage_id = omni.usd.get_context().get_stage_id()

            self._tilesets.append(
                self._cesium_omniverse_interface.addTilesetIon(
                    stage_id,
                    1387142,
                    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyMjRhYzI0Yi1kNWEwLTQ4ZWYtYjdmZC1hY2JmYWIzYmFiMGUiLCJpZCI6NDQsImlhdCI6MTY2NzQ4OTg0N30.du0tvWptgLWsvM1Gnbv3Zw_pDAOILg1Wr6s2sgK-qlM",
                )
            )

        def add_cesium_world_terrain():
            """Adds the standard Cesium World Terrain to the stage."""

            # Cesium HQ
            self._cesium_omniverse_interface.setGeoreferenceOrigin(-75.1564977, 39.9501464, 150.0)

            stage_id = omni.usd.get_context().get_stage_id()

            tileset_id = self._cesium_omniverse_interface.addTilesetIon(
                stage_id,
                1,
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyMjRhYzI0Yi1kNWEwLTQ4ZWYtYjdmZC1hY2JmYWIzYmFiMGUiLCJpZCI6NDQsImlhdCI6MTY2NzQ4OTg0N30.du0tvWptgLWsvM1Gnbv3Zw_pDAOILg1Wr6s2sgK-qlM",
            )

            self._tilesets.append(tileset_id)

            self._cesium_omniverse_interface.addIonRasterOverlay(
                tileset_id,
                "Layer",
                3954,
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyMjRhYzI0Yi1kNWEwLTQ4ZWYtYjdmZC1hY2JmYWIzYmFiMGUiLCJpZCI6NDQsImlhdCI6MTY2NzQ4OTg0N30.du0tvWptgLWsvM1Gnbv3Zw_pDAOILg1Wr6s2sgK-qlM",
            )

        def add_bing_maps_terrain():
            """Adds the Bing Maps & Cesium Terrain to the stage."""

            # Cesium HQ
            self._cesium_omniverse_interface.setGeoreferenceOrigin(-75.1564977, 39.9501464, 150.0)

            stage_id = omni.usd.get_context().get_stage_id()

            tileset_id = self._cesium_omniverse_interface.addTilesetIon(
                stage_id,
                1,
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyMjRhYzI0Yi1kNWEwLTQ4ZWYtYjdmZC1hY2JmYWIzYmFiMGUiLCJpZCI6NDQsImlhdCI6MTY2NzQ4OTg0N30.du0tvWptgLWsvM1Gnbv3Zw_pDAOILg1Wr6s2sgK-qlM",
            )

            self._tilesets.append(tileset_id)

            self._cesium_omniverse_interface.addIonRasterOverlay(
                tileset_id,
                "Layer",
                2,
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyMjRhYzI0Yi1kNWEwLTQ4ZWYtYjdmZC1hY2JmYWIzYmFiMGUiLCJpZCI6NDQsImlhdCI6MTY2NzQ4OTg0N30.du0tvWptgLWsvM1Gnbv3Zw_pDAOILg1Wr6s2sgK-qlM",
            )

        def create_tileset(tileset=Tileset.CESIUM_WORLD_TERRAIN):
            """Creates the desired tileset on the stage.

            Parameters:
                tileset (Tileset): The desired tileset specified using the Tileset enumeration.
            """

            self.destroy_subscription()

            self.update_far_plane()

            # TODO: Eventually we need a real way to do this.
            if tileset is Tileset.CAPE_CANAVERAL:
                add_maxar_3d_surface_model()
            elif tileset is Tileset.BING_MAPS:
                add_bing_maps_terrain()
            else:  # Terrain is Cesium World Terrain
                add_cesium_world_terrain()

        with ui.VStack():
            ui.Button("Update Frame", clicked_fn=lambda: start_update_frame())
            ui.Button("Stop Update Frame", clicked_fn=lambda: stop_update_frame())
            ui.Button(
                "Create Cesium World Terrain Tileset", clicked_fn=lambda: create_tileset(Tileset.CESIUM_WORLD_TERRAIN)
            )
            ui.Button("Create Bing Maps Tileset", clicked_fn=lambda: create_tileset(Tileset.BING_MAPS))
            ui.Button("Create Cape Canaveral Tileset", clicked_fn=lambda: create_tileset(Tileset.CAPE_CANAVERAL))

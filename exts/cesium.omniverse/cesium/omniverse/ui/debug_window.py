import carb.events
import logging
from typing import Optional, List
import omni.kit.app as app
import omni.ui as ui
import omni.usd
from omni.kit.viewport.utility import get_active_viewport
from pxr import Usd, UsdGeom, Gf
from .statistics_widget import CesiumOmniverseStatisticsWidget
from ..bindings import ICesiumOmniverseInterface
from ..usdUtils import remove_tileset


class CesiumOmniverseDebugWindow(ui.Window):
    WINDOW_NAME = "Cesium Debugging"
    MENU_PATH = f"Window/Cesium/{WINDOW_NAME}"

    _logger: logging.Logger
    _cesium_omniverse_interface: Optional[ICesiumOmniverseInterface] = None

    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, title: str, **kwargs):
        super().__init__(title, **kwargs)
        self._subscriptions: List[carb.events.ISubscription] = []

        self._logger = logging.getLogger(__name__)
        self._cesium_omniverse_interface = cesium_omniverse_interface
        self._cesium_message_field: ui.SimpleStringModel = ui.SimpleStringModel("")
        self._statistics_widget: Optional[CesiumOmniverseStatisticsWidget] = None

        # Set the function that is called to build widgets when the window is visible
        self.frame.set_build_fn(self._build_fn)

    def destroy(self):
        if self._statistics_widget is not None:
            self._statistics_widget.destroy()
            self._statistics_widget = None

        # It will destroy all the children
        super().destroy()

    def __del__(self):
        self.destroy()

    @staticmethod
    def show_window():
        ui.Workspace.show_window(CesiumOmniverseDebugWindow.WINDOW_NAME)

    def _on_update_frame(self, _):
        stage = omni.usd.get_context().get_stage()
        viewport = get_active_viewport()
        camera_path = viewport.get_active_camera()
        camera = UsdGeom.Camera.Get(stage, camera_path)
        xform = UsdGeom.Xformable(camera)
        t = Usd.TimeCode.Default()  # The time at which we compute the bounding box
        world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(t)
        translation: Gf.Vec3d = world_transform.ExtractTranslation()
        up_vector = Gf.Vec3f(world_transform[1][0], world_transform[1][1], world_transform[1][2])

        self._cesium_omniverse_interface.run_kernel_on_point_cloud(
            translation[0], translation[1], translation[2],
            up_vector[0], up_vector[1], up_vector[2])

    def _build_fn(self):
        """Builds out the UI buttons and their handlers."""

        def remove_all_tilesets():
            """Removes all tilesets from the stage."""

            tileset_paths = self._cesium_omniverse_interface.get_all_tileset_paths()

            for tileset_path in tileset_paths:
                remove_tileset(tileset_path)

        def reload_all_tilesets():
            """Reloads all tilesets."""

            tileset_paths = self._cesium_omniverse_interface.get_all_tileset_paths()

            for tileset_path in tileset_paths:
                self._cesium_omniverse_interface.reload_tileset(tileset_path)

        def print_fabric_stage():
            """Prints the contents of the Fabric stage to a text field."""

            fabric_stage = self._cesium_omniverse_interface.print_fabric_stage()
            self._cesium_message_field.set_value(fabric_stage)

        def create_test_point_cloud():
            self._cesium_omniverse_interface.create_test_point_cloud()

        def run_kernel_on_point_cloud():
            update_stream = app.get_app().get_update_event_stream()
            self._subscriptions.append(
                update_stream.create_subscription_to_pop(
                    self._on_update_frame, name="cesium.omniverse.viewport.ON_UPDATE_FRAME"
                )
            )

        with ui.VStack(spacing=10):
            with ui.VStack():
                ui.Button("Remove all Tilesets", height=20, clicked_fn=remove_all_tilesets)
                ui.Button("Reload all Tilesets", height=20, clicked_fn=reload_all_tilesets)
                ui.Button("Print Fabric stage", height=20, clicked_fn=print_fabric_stage)
                ui.Button("Load test point cloud", height=20, clicked_fn=create_test_point_cloud)
                ui.Button("Run kernel on point cloud (in update loop)", height=20, clicked_fn=run_kernel_on_point_cloud)
                ui.StringField(self._cesium_message_field, height=100, multiline=True, read_only=True)
            self._statistics_widget = CesiumOmniverseStatisticsWidget(self._cesium_omniverse_interface)

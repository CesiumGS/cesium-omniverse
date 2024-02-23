from .bindings import acquire_cesium_omniverse_interface, release_cesium_omniverse_interface, Viewport
from .ui.add_menu_controller import CesiumAddMenuController
from .install import perform_vendor_install
from .utils import wait_n_frames, dock_window_async, perform_action_after_n_frames_async
from .usdUtils import (
    add_tileset_ion,
    add_raster_overlay_ion,
    add_cartographic_polygon,
    get_or_create_cesium_data,
    get_or_create_cesium_georeference,
)
from .ui.asset_window import CesiumOmniverseAssetWindow
from .ui.debug_window import CesiumOmniverseDebugWindow
from .ui.main_window import CesiumOmniverseMainWindow
from .ui.settings_window import CesiumOmniverseSettingsWindow
from .ui.credits_viewport_frame import CesiumCreditsViewportFrame
from .ui.fabric_modal import CesiumFabricModal
from .models import AssetToAdd, RasterOverlayToAdd
from .ui import CesiumAttributesWidgetController
import asyncio
from functools import partial
import logging
import carb.events
import carb.settings as omni_settings
import omni.ext
import omni.kit.app as omni_app
import omni.kit.ui
import omni.kit.pipapi
from omni.kit.viewport.window import get_viewport_window_instances
import omni.ui as ui
import omni.usd
import os
from typing import List, Optional, Callable
from .ui.credits_viewport_controller import CreditsViewportController
from cesium.usd.plugins.CesiumUsdSchemas import Data as CesiumData, IonServer as CesiumIonServer
from omni.kit.capture.viewport import CaptureExtension

CESIUM_DATA_PRIM_PATH = "/Cesium"

cesium_extension_location = os.path.join(os.path.dirname(__file__), "../../")


class CesiumOmniverseExtension(omni.ext.IExt):
    @staticmethod
    def _set_menu(path, value):
        # Set the menu to create this window on and off
        editor_menu = omni.kit.ui.get_editor_menu()
        if editor_menu:
            editor_menu.set_value(path, value)

    def __init__(self) -> None:
        super().__init__()

        self._main_window: Optional[CesiumOmniverseMainWindow] = None
        self._asset_window: Optional[CesiumOmniverseAssetWindow] = None
        self._debug_window: Optional[CesiumOmniverseDebugWindow] = None
        self._settings_window: Optional[CesiumOmniverseSettingsWindow] = None
        self._credits_viewport_frames: List[CesiumCreditsViewportFrame] = []
        self._on_stage_subscription: Optional[carb.events.ISubscription] = None
        self._on_update_subscription: Optional[carb.events.ISubscription] = None
        self._show_asset_window_subscription: Optional[carb.events.ISubscription] = None
        self._token_set_subscription: Optional[carb.events.ISubscription] = None
        self._add_ion_asset_subscription: Optional[carb.events.ISubscription] = None
        self._add_blank_asset_subscription: Optional[carb.events.ISubscription] = None
        self._add_raster_overlay_subscription: Optional[carb.events.ISubscription] = None
        self._add_cartographic_polygon_subscription: Optional[carb.events.ISubscription] = None
        self._assets_to_add_after_token_set: List[AssetToAdd] = []
        self._raster_overlay_to_add_after_token_set: List[RasterOverlayToAdd] = []
        self._adding_assets = False
        self._attributes_widget_controller: Optional[CesiumAttributesWidgetController] = None
        self._credits_viewport_controller: Optional[CreditsViewportController] = None
        self._add_menu_controller: Optional[CesiumAddMenuController] = None
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._menus = []
        self._num_credits_viewport_frames: int = 0
        self._capture_instance = None

        perform_vendor_install()

    def on_startup(self):
        # The ability to show up the window if the system requires it. We use it in QuickLayout.
        ui.Workspace.set_show_window_fn(CesiumOmniverseMainWindow.WINDOW_NAME, partial(self.show_main_window, None))
        ui.Workspace.set_show_window_fn(
            CesiumOmniverseAssetWindow.WINDOW_NAME, partial(self.show_assets_window, None)
        )
        ui.Workspace.set_show_window_fn(CesiumOmniverseDebugWindow.WINDOW_NAME, partial(self.show_debug_window, None))
        ui.Workspace.set_show_window_fn(
            CesiumOmniverseSettingsWindow.WINDOW_NAME, partial(self.show_settings_window, None)
        )

        settings = omni_settings.get_settings()
        show_on_startup = settings.get_as_bool("/exts/cesium.omniverse/showOnStartup")

        self._add_to_menu(CesiumOmniverseMainWindow.MENU_PATH, self.show_main_window, show_on_startup)
        self._add_to_menu(CesiumOmniverseAssetWindow.MENU_PATH, self.show_assets_window, False)
        self._add_to_menu(CesiumOmniverseDebugWindow.MENU_PATH, self.show_debug_window, False)
        self._add_to_menu(CesiumOmniverseSettingsWindow.MENU_PATH, self.show_settings_window, False)

        self._logger.info("CesiumOmniverse startup")

        # Acquire the Cesium Omniverse interface.
        global _cesium_omniverse_interface
        _cesium_omniverse_interface = acquire_cesium_omniverse_interface()
        _cesium_omniverse_interface.on_startup(cesium_extension_location)

        settings.set("/rtx/hydra/TBNFrameMode", 1)

        # Allow material graph to find cesium mdl exports
        mdl_custom_paths_name = "materialConfig/searchPaths/custom"
        mdl_user_allow_list_name = "materialConfig/materialGraph/userAllowList"
        mdl_renderer_custom_paths_name = "/renderer/mdl/searchPaths/custom"

        cesium_mdl_search_path = os.path.join(cesium_extension_location, "mdl")
        cesium_mdl_name = "cesium.mdl"

        mdl_custom_paths = settings.get(mdl_custom_paths_name) or []
        mdl_user_allow_list = settings.get(mdl_user_allow_list_name) or []

        mdl_custom_paths.append(cesium_mdl_search_path)
        mdl_user_allow_list.append(cesium_mdl_name)

        mdl_renderer_custom_paths = settings.get_as_string(mdl_renderer_custom_paths_name)
        mdl_renderer_custom_paths_sep = "" if mdl_renderer_custom_paths == "" else ";"
        mdl_renderer_custom_paths = mdl_renderer_custom_paths + mdl_renderer_custom_paths_sep + cesium_mdl_search_path

        settings.set_string_array(mdl_custom_paths_name, mdl_custom_paths)
        settings.set_string_array(mdl_user_allow_list_name, mdl_user_allow_list)
        settings.set_string(mdl_renderer_custom_paths_name, mdl_renderer_custom_paths)

        # Show the window. It will call `self.show_window`
        if show_on_startup:
            asyncio.ensure_future(perform_action_after_n_frames_async(15, CesiumOmniverseExtension._open_window))

        self._credits_viewport_controller = CreditsViewportController(_cesium_omniverse_interface)

        self._add_menu_controller = CesiumAddMenuController(_cesium_omniverse_interface)

        # Subscribe to stage event stream
        usd_context = omni.usd.get_context()
        if usd_context.get_stage_state() == omni.usd.StageState.OPENED:
            _cesium_omniverse_interface.on_stage_change(usd_context.get_stage_id())

        self._on_stage_subscription = usd_context.get_stage_event_stream().create_subscription_to_pop(
            self._on_stage_event, name="cesium.omniverse.ON_STAGE_EVENT"
        )

        self._on_update_subscription = (
            omni_app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_update_frame, name="cesium.omniverse.extension.ON_UPDATE_FRAME")
        )

        bus = omni_app.get_app().get_message_bus_event_stream()
        show_asset_window_event = carb.events.type_from_string("cesium.omniverse.SHOW_ASSET_WINDOW")
        self._show_asset_window_subscription = bus.create_subscription_to_pop_by_type(
            show_asset_window_event, self._on_show_asset_window_event
        )

        token_set_event = carb.events.type_from_string("cesium.omniverse.SET_DEFAULT_TOKEN_SUCCESS")
        self._token_set_subscription = bus.create_subscription_to_pop_by_type(token_set_event, self._on_token_set)

        add_ion_asset_event = carb.events.type_from_string("cesium.omniverse.ADD_ION_ASSET")
        self._add_ion_asset_subscription = bus.create_subscription_to_pop_by_type(
            add_ion_asset_event, self._on_add_ion_asset_event
        )

        add_blank_asset_event = carb.events.type_from_string("cesium.omniverse.ADD_BLANK_ASSET")
        self._add_blank_asset_subscription = bus.create_subscription_to_pop_by_type(
            add_blank_asset_event, self._on_add_blank_asset_event
        )

        add_cartographic_polygon_event = carb.events.type_from_string("cesium.omniverse.ADD_CARTOGRAPHIC_POLYGON")
        self._add_cartographic_polygon_subscription = bus.create_subscription_to_pop_by_type(
            add_cartographic_polygon_event, self._on_add_cartographic_polygon_event
        )

        add_raster_overlay_event = carb.events.type_from_string("cesium.omniverse.ADD_RASTER_OVERLAY")
        self._add_raster_overlay_subscription = bus.create_subscription_to_pop_by_type(
            add_raster_overlay_event, self._on_add_raster_overlay_to_tileset
        )

        self._capture_instance = CaptureExtension.get_instance()

    def on_shutdown(self):
        self._menus.clear()

        if self._main_window is not None:
            self._main_window.destroy()
            self._main_window = None

        if self._asset_window is not None:
            self._asset_window.destroy()
            self._asset_window = None

        if self._debug_window is not None:
            self._debug_window.destroy()
            self._debug_window = None

        if self._settings_window is not None:
            self._settings_window.destroy()
            self._settings_window = None

        if self._credits_viewport_controller is not None:
            self._credits_viewport_controller.destroy()
            self._credits_viewport_controller = None

        # Deregister the function that shows the window from omni.ui
        ui.Workspace.set_show_window_fn(CesiumOmniverseMainWindow.WINDOW_NAME, None)
        ui.Workspace.set_show_window_fn(CesiumOmniverseAssetWindow.WINDOW_NAME, None)
        ui.Workspace.set_show_window_fn(CesiumOmniverseDebugWindow.WINDOW_NAME, None)
        ui.Workspace.set_show_window_fn(CesiumOmniverseSettingsWindow.WINDOW_NAME, None)

        if self._on_stage_subscription is not None:
            self._on_stage_subscription.unsubscribe()
            self._on_stage_subscription = None

        if self._on_update_subscription is not None:
            self._on_update_subscription.unsubscribe()
            self._on_update_subscription = None

        if self._token_set_subscription is not None:
            self._token_set_subscription.unsubscribe()
            self._token_set_subscription = None

        if self._add_ion_asset_subscription is not None:
            self._add_ion_asset_subscription.unsubscribe()
            self._add_ion_asset_subscription = None

        if self._add_blank_asset_subscription is not None:
            self._add_blank_asset_subscription.unsubscribe()
            self._add_blank_asset_subscription = None

        if self._add_raster_overlay_subscription is not None:
            self._add_raster_overlay_subscription.unsubscribe()
            self._add_raster_overlay_subscription = None

        if self._add_cartographic_polygon_subscription is not None:
            self._add_cartographic_polygon_subscription.unsubscribe()
            self._add_cartographic_polygon_subscription = None

        if self._show_asset_window_subscription is not None:
            self._show_asset_window_subscription.unsubscribe()
            self._show_asset_window_subscription = None

        if self._attributes_widget_controller is not None:
            self._attributes_widget_controller.destroy()
            self._attributes_widget_controller = None

        if self._add_menu_controller is not None:
            self._add_menu_controller.destroy()
            self._add_menu_controller = None

        self._capture_instance = None

        self._destroy_credits_viewport_frames()

        self._logger.info("CesiumOmniverse shutdown")

        # Release the Cesium Omniverse interface.
        _cesium_omniverse_interface.on_shutdown()
        release_cesium_omniverse_interface(_cesium_omniverse_interface)

    def _on_update_frame(self, _):
        if omni.usd.get_context().get_stage_state() != omni.usd.StageState.OPENED:
            return

        viewports = []
        for instance in get_viewport_window_instances():
            viewport_api = instance.viewport_api
            viewport = Viewport()
            viewport.viewMatrix = viewport_api.view
            viewport.projMatrix = viewport_api.projection
            viewport.width = float(viewport_api.resolution[0])
            viewport.height = float(viewport_api.resolution[1])
            viewports.append(viewport)

        if len(viewports) != self._num_credits_viewport_frames:
            self._setup_credits_viewport_frames()
            self._num_credits_viewport_frames = len(viewports)

        wait_for_loading_tiles = (
            self._capture_instance.progress.capture_status == omni.kit.capture.viewport.CaptureStatus.CAPTURING
        )
        _cesium_omniverse_interface.on_update_frame(viewports, wait_for_loading_tiles)

    def _on_stage_event(self, event):
        if _cesium_omniverse_interface is None:
            return

        if event.type == int(omni.usd.StageEventType.OPENED):
            _cesium_omniverse_interface.on_stage_change(omni.usd.get_context().get_stage_id())
            self._attributes_widget_controller = CesiumAttributesWidgetController(_cesium_omniverse_interface)

            # Show Fabric modal if Fabric is disabled.
            fabric_enabled = omni_settings.get_settings().get_as_bool("/app/useFabricSceneDelegate")
            if not fabric_enabled:
                asyncio.ensure_future(perform_action_after_n_frames_async(15, CesiumOmniverseExtension._open_modal))

            get_or_create_cesium_data()
            get_or_create_cesium_georeference()

            self._setup_ion_server_prims()
        elif event.type == int(omni.usd.StageEventType.CLOSED):
            _cesium_omniverse_interface.on_stage_change(0)
            if self._attributes_widget_controller is not None:
                self._attributes_widget_controller.destroy()
                self._attributes_widget_controller = None

    def _on_show_asset_window_event(self, _):
        self.do_show_assets_window()

    def _on_token_set(self, _: carb.events.IEvent):
        if self._adding_assets:
            return

        self._adding_assets = True

        for asset in self._assets_to_add_after_token_set:
            self._add_ion_assets(asset)
        self._assets_to_add_after_token_set.clear()

        for raster_overlay in self._raster_overlay_to_add_after_token_set:
            self._add_raster_overlay_to_tileset(raster_overlay)
        self._raster_overlay_to_add_after_token_set.clear()

        self._adding_assets = False

    def _on_add_ion_asset_event(self, event: carb.events.IEvent):
        asset_to_add = AssetToAdd.from_event(event)

        self._add_ion_assets(asset_to_add)

    def _on_add_blank_asset_event(self, event: carb.events.IEvent):
        asset_to_add = AssetToAdd.from_event(event)

        self._add_ion_assets(asset_to_add, skip_ion_checks=True)

    def _on_add_cartographic_polygon_event(self, event: carb.events.IEvent):
        self._add_cartographic_polygon_assets()

    def _add_ion_assets(self, asset_to_add: Optional[AssetToAdd], skip_ion_checks=False):
        if asset_to_add is None:
            self._logger.warning("Insufficient information to add asset.")
            return

        if not skip_ion_checks:
            session = _cesium_omniverse_interface.get_session()

            if not session.is_connected():
                self._logger.warning("Must be logged in to add ion asset.")
                return

            if not _cesium_omniverse_interface.is_default_token_set():
                bus = omni_app.get_app().get_message_bus_event_stream()
                show_token_window_event = carb.events.type_from_string("cesium.omniverse.SHOW_TOKEN_WINDOW")
                bus.push(show_token_window_event)
                self._assets_to_add_after_token_set.append(asset_to_add)
                return

        if asset_to_add.raster_overlay_name is not None and asset_to_add.raster_overlay_ion_asset_id is not None:
            tileset_path = add_tileset_ion(asset_to_add.tileset_name, asset_to_add.tileset_ion_asset_id)
            add_raster_overlay_ion(
                tileset_path, asset_to_add.raster_overlay_name, asset_to_add.raster_overlay_ion_asset_id
            )
        else:
            tileset_path = add_tileset_ion(asset_to_add.tileset_name, asset_to_add.tileset_ion_asset_id)

        if tileset_path == "":
            self._logger.warning("Error adding tileset and raster overlay to stage")

    def _add_cartographic_polygon_assets(self):
        add_cartographic_polygon()

    def _on_add_raster_overlay_to_tileset(self, event: carb.events.IEvent):
        raster_overlay_to_add = RasterOverlayToAdd.from_event(event)

        if raster_overlay_to_add is None:
            self._logger.warning("Insufficient information to add raster overlay.")

        self._add_raster_overlay_to_tileset(raster_overlay_to_add)

    def _add_raster_overlay_to_tileset(self, raster_overlay_to_add: RasterOverlayToAdd):
        session = _cesium_omniverse_interface.get_session()

        if not session.is_connected():
            self._logger.warning("Must be logged in to add ion asset.")
            return

        if not _cesium_omniverse_interface.is_default_token_set():
            bus = omni_app.get_app().get_message_bus_event_stream()
            show_token_window_event = carb.events.type_from_string("cesium.omniverse.SHOW_TOKEN_WINDOW")
            bus.push(show_token_window_event)
            self._raster_overlay_to_add_after_token_set.append(raster_overlay_to_add)
            return

        add_raster_overlay_ion(
            raster_overlay_to_add.tileset_path,
            raster_overlay_to_add.raster_overlay_name,
            raster_overlay_to_add.raster_overlay_ion_asset_id,
        )
        _cesium_omniverse_interface.reload_tileset(raster_overlay_to_add.tileset_path)

    def _add_to_menu(self, path, callback: Callable[[bool], None], show_on_startup):
        editor_menu = omni.kit.ui.get_editor_menu()

        if editor_menu:
            self._menus.append(editor_menu.add_item(path, callback, toggle=True, value=show_on_startup))

    async def _destroy_window_async(self, path):
        # Wait one frame, this is due to the one frame defer in Window::_moveToMainOSWindow()
        await wait_n_frames(1)

        if path is CesiumOmniverseMainWindow.MENU_PATH:
            if self._main_window is not None:
                self._main_window.destroy()
            self._main_window = None
        elif path is CesiumOmniverseAssetWindow.MENU_PATH:
            if self._asset_window is not None:
                self._asset_window.destroy()
            self._asset_window = None
        elif path is CesiumOmniverseDebugWindow.MENU_PATH:
            if self._debug_window is not None:
                self._debug_window.destroy()
            self._debug_window = None
        elif path is CesiumOmniverseSettingsWindow.MENU_PATH:
            if self._settings_window is not None:
                self._settings_window.destroy()
            self._settings_window = None

    def _visibility_changed_fn(self, path, visible):
        # Called when the user pressed "X"
        self._set_menu(path, visible)
        if not visible:
            # Destroy the window, since we are creating new window in show_window
            asyncio.ensure_future(self._destroy_window_async(path))

    def show_main_window(self, _menu, value):
        if _cesium_omniverse_interface is None:
            logging.error("Cesium Omniverse Interface is not set.")
            return

        if value:
            self._main_window = CesiumOmniverseMainWindow(_cesium_omniverse_interface, width=300, height=400)
            self._main_window.set_visibility_changed_fn(
                partial(self._visibility_changed_fn, CesiumOmniverseMainWindow.MENU_PATH)
            )
            asyncio.ensure_future(dock_window_async(self._main_window))
        elif self._main_window is not None:
            self._main_window.visible = False

    def do_show_assets_window(self):
        if self._asset_window:
            self._asset_window.focus()
            return

        self._asset_window = CesiumOmniverseAssetWindow(_cesium_omniverse_interface, width=700, height=300)
        self._asset_window.set_visibility_changed_fn(
            partial(self._visibility_changed_fn, CesiumOmniverseAssetWindow.MENU_PATH)
        )
        asyncio.ensure_future(dock_window_async(self._asset_window, "Content"))

    def show_assets_window(self, _menu, value):
        if _cesium_omniverse_interface is None:
            logging.error("Cesium Omniverse Interface is not set.")
            return

        if value:
            self.do_show_assets_window()
        elif self._asset_window is not None:
            self._asset_window.visible = False

    def show_debug_window(self, _menu, value):
        if _cesium_omniverse_interface is None:
            logging.error("Cesium Omniverse Interface is not set.")
            return

        if value:
            self._debug_window = CesiumOmniverseDebugWindow(
                _cesium_omniverse_interface, CesiumOmniverseDebugWindow.WINDOW_NAME, width=300, height=365
            )
            self._debug_window.set_visibility_changed_fn(
                partial(self._visibility_changed_fn, CesiumOmniverseDebugWindow.MENU_PATH)
            )
            asyncio.ensure_future(dock_window_async(self._debug_window))
        elif self._debug_window is not None:
            self._debug_window.visible = False

    def show_settings_window(self, _menu, value):
        if _cesium_omniverse_interface is None:
            logging.error("Cesium Omniverse Interface is not set.")
            return

        if value:
            self._settings_window = CesiumOmniverseSettingsWindow(
                _cesium_omniverse_interface, CesiumOmniverseSettingsWindow.WINDOW_NAME, width=300, height=365
            )
            self._settings_window.set_visibility_changed_fn(
                partial(self._visibility_changed_fn, CesiumOmniverseSettingsWindow.MENU_PATH)
            )
            asyncio.ensure_future(dock_window_async(self._settings_window))
        elif self._settings_window is not None:
            self._settings_window.visible = False

    def _setup_credits_viewport_frames(self):
        self._destroy_credits_viewport_frames()
        self._credits_viewport_frames = [
            CesiumCreditsViewportFrame(_cesium_omniverse_interface, i) for i in get_viewport_window_instances()
        ]
        if self._credits_viewport_controller is not None:
            self._credits_viewport_controller.broadcast_credits()

    def _destroy_credits_viewport_frames(self):
        for credits_viewport_frame in self._credits_viewport_frames:
            credits_viewport_frame.destroy()
        self._credits_viewport_frames.clear()

    @staticmethod
    def _open_window():
        ui.Workspace.show_window(CesiumOmniverseMainWindow.WINDOW_NAME)

    @staticmethod
    def _open_modal():
        CesiumFabricModal()

    def _setup_ion_server_prims(self):
        # TODO: Move a lot of this to usdUtils.py
        stage = omni.usd.get_context().get_stage()
        server_prims: List[CesiumIonServer] = [x for x in stage.Traverse() if x.IsA(CesiumIonServer)]

        if len(server_prims) < 1:
            # If we have no ion server prims, lets add a default one for the official ion servers.
            path = "/CesiumServers/IonOfficial"
            prim: CesiumIonServer = CesiumIonServer.Define(stage, path)
            prim.GetDisplayNameAttr().Set("ion.cesium.com")
            prim.GetIonServerUrlAttr().Set("https://ion.cesium.com/")
            prim.GetIonServerApiUrlAttr().Set("https://api.cesium.com/")
            prim.GetIonServerApplicationIdAttr().Set(413)

            data_prim: CesiumData = CesiumData.Get(stage, CESIUM_DATA_PRIM_PATH)
            data_prim.GetSelectedIonServerRel().AddTarget(path)

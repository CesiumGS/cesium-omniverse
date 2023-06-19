from functools import partial
import asyncio
import time
from typing import Callable, Optional
import logging
import carb.events
import omni.ext
import omni.ui as ui
import omni.usd
import omni.kit.app as app
import omni.kit.ui
from .performance_window import CesiumPerformanceWindow
from cesium.omniverse.bindings import acquire_cesium_omniverse_interface, release_cesium_omniverse_interface
from cesium.omniverse.utils import wait_n_frames, dock_window_async
from cesium.usd.plugins.CesiumUsdSchemas import TilesetAPI as CesiumTilesetAPI, Data as CesiumData

ION_ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyZTA0MDlmYi01Y2RhLTQ0MjQtYjBlOS1kMmZhMzQ0OWRkNGYiLCJpZCI6MjU5LCJpYXQiOjE2ODU2MzExMTF9.y2CrqatkaHKHcj6NIDJ8ioll-tnOi-2CblnzI6iUays"
GOOGLE_3D_TILES_ACCESS_TOKEN = "AIzaSyC2PMYr_ZaMJT5DdZ8WJNYMwB0lDyvx5q8"


class CesiumPerformanceExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()

        self._logger = logging.getLogger(__name__)

        self._performance_window: Optional[CesiumPerformanceWindow] = None
        self._view_new_york_city_subscription: Optional[carb.events.ISubscription] = None
        self._view_grand_canyon_subscription: Optional[carb.events.ISubscription] = None
        self._view_tour_subscription: Optional[carb.events.ISubscription] = None
        self._tileset_loaded_subscription: Optional[carb.events.ISubscription] = None

        self._tileset_path: Optional[str] = None
        self._active: bool = False
        self._start_time: float = 0.0

    def on_startup(self):
        global _cesium_omniverse_interface
        _cesium_omniverse_interface = acquire_cesium_omniverse_interface()

        self._setup_menus()
        self._show_and_dock_startup_windows()

        bus = app.get_app().get_message_bus_event_stream()
        view_new_york_city_event = carb.events.type_from_string("cesium.performance.VIEW_NEW_YORK_CITY")
        self._view_new_york_city_subscription = bus.create_subscription_to_pop_by_type(
            view_new_york_city_event, self._view_new_york_city
        )

        view_grand_canyon_event = carb.events.type_from_string("cesium.performance.VIEW_GRAND_CANYON")
        self._view_grand_canyon_subscription = bus.create_subscription_to_pop_by_type(
            view_grand_canyon_event, self._view_grand_canyon
        )

        view_tour_event = carb.events.type_from_string("cesium.performance.VIEW_TOUR")
        self._view_tour_subscription = bus.create_subscription_to_pop_by_type(view_tour_event, self._view_tour)

        stop_event = carb.events.type_from_string("cesium.performance.STOP")
        self._stop_subscription = bus.create_subscription_to_pop_by_type(stop_event, self._on_stop)

        update_stream = app.get_app().get_update_event_stream()
        self._update_frame_subscription = update_stream.create_subscription_to_pop(
            self._on_update_frame, name="cesium.performance.ON_UPDATE_FRAME"
        )

    def on_shutdown(self):
        self._clear_scene()

        if self._view_new_york_city_subscription is not None:
            self._view_new_york_city_subscription.unsubscribe()
            self._view_new_york_city_subscription = None

        if self._view_grand_canyon_subscription is not None:
            self._view_grand_canyon_subscription.unsubscribe()
            self._view_grand_canyon_subscription = None

        if self._view_tour_subscription is not None:
            self._view_tour_subscription.unsubscribe()
            self._view_tour_subscription = None

        if self._stop_subscription is not None:
            self._stop_subscription.unsubscribe()
            self._stop_subscription = None

        if self._update_frame_subscription is not None:
            self._update_frame_subscription.unsubscribe()
            self._update_frame_subscription = None

        self._destroy_performance_window()

        release_cesium_omniverse_interface(_cesium_omniverse_interface)

    def _setup_menus(self):
        ui.Workspace.set_show_window_fn(
            CesiumPerformanceWindow.WINDOW_NAME, partial(self._show_performance_window, None)
        )

        editor_menu = omni.kit.ui.get_editor_menu()

        if editor_menu:
            editor_menu.add_item(
                CesiumPerformanceWindow.MENU_PATH, self._show_performance_window, toggle=True, value=True
            )

    def _show_and_dock_startup_windows(self):
        ui.Workspace.show_window(CesiumPerformanceWindow.WINDOW_NAME)

        asyncio.ensure_future(dock_window_async(self._performance_window, target="Property"))

    def _destroy_performance_window(self):
        if self._performance_window is not None:
            self._performance_window.destroy()
        self._performance_window = None

    async def _destroy_window_async(self, path):
        # Wait one frame, this is due to the one frame defer in Window::_moveToMainOSWindow()
        await wait_n_frames(1)

        if path is CesiumPerformanceWindow.MENU_PATH:
            self._destroy_performance_window()

    def _visibility_changed_fn(self, path, visible):
        editor_menu = omni.kit.ui.get_editor_menu()
        if editor_menu:
            editor_menu.set_value(path, visible)
        if not visible:
            asyncio.ensure_future(self._destroy_window_async(path))

    def _show_performance_window(self, _menu, value):
        if value:
            self._performance_window = CesiumPerformanceWindow(_cesium_omniverse_interface, width=300, height=400)
            self._performance_window.set_visibility_changed_fn(
                partial(self._visibility_changed_fn, CesiumPerformanceWindow.MENU_PATH)
            )
        elif self._performance_window is not None:
            self._performance_window.visible = False

    def _on_update_frame(self, _e: carb.events.IEvent):
        if self._active is True:
            duration = self._get_duration()
            self._update_duration_ui(duration)

    def _view_new_york_city(self, _e: carb.events.IEvent):
        self._logger.warning("View New York City")

        self._clear_scene()

        _cesium_omniverse_interface.set_georeference_origin(-74.0, 40.69, 50)

        tileset_path = _cesium_omniverse_interface.add_tileset_ion("Cesium_World_Terrain", 1, ION_ACCESS_TOKEN)

        self._load_tileset(tileset_path, self._tileset_loaded)

    def _view_grand_canyon(self, _e: carb.events.IEvent):
        self._logger.warning("View Grand Canyon")

    def _view_tour(self, _e: carb.events.IEvent):
        self._logger.warning("View Tour")

        self._clear_scene()

        def tour_stop_0():
            _cesium_omniverse_interface.set_georeference_origin(-74.0, 40.69, 50)

        def tour_stop_1():
            _cesium_omniverse_interface.set_georeference_origin(2.349, 48.86, 100)

        def tour_stop_2():
            _cesium_omniverse_interface.set_georeference_origin(-157.86, 21.31, 10)

        tour_stops = [tour_stop_0, tour_stop_1, tour_stop_2]
        current_stop = 0

        def tileset_loaded(_e: carb.events.IEvent):
            nonlocal current_stop

            duration = self._get_duration()
            self._logger.warning("Tour stop {} loaded in {} seconds".format(current_stop, duration))

            if current_stop == len(tour_stops) - 1:
                self._tileset_loaded(_e)
            else:
                current_stop += 1
                tour_stops[current_stop]()

        tour_stops[0]()

        tileset_path = _cesium_omniverse_interface.add_tileset_ion("Cesium_World_Terrain", 1, ION_ACCESS_TOKEN)

        self._load_tileset(tileset_path, tileset_loaded)

    def _load_tileset(self, tileset_path: str, tileset_loaded: Callable):
        stage = omni.usd.get_context().get_stage()

        tileset_prim = CesiumTilesetAPI.Get(stage, tileset_path)
        if not tileset_prim.GetPrim().IsValid():
            self._logger.error("Can't run performance test: tileset prim is not valid")
            return

        cesium_prim = CesiumData.Get(stage, "/Cesium")
        if not cesium_prim.GetPrim().IsValid():
            self._logger.error("Can't run performance test: cesium prim is not valid")
            return

        if self._performance_window is None:
            self._logger.error("Can't run performance test: performance window is None")
            return

        bus = app.get_app().get_message_bus_event_stream()
        tileset_loaded_event = carb.events.type_from_string("cesium.omniverse.TILESET_LOADED")
        self._tileset_loaded_subscription = bus.create_subscription_to_pop_by_type(
            tileset_loaded_event, tileset_loaded
        )

        random_colors = self._performance_window.get_random_colors()
        forbid_holes = self._performance_window.get_forbid_holes()
        frustum_culling = self._performance_window.get_frustum_culling()

        cesium_prim.GetDebugRandomColorsAttr().Set(random_colors)
        tileset_prim.GetForbidHolesAttr().Set(forbid_holes)
        tileset_prim.GetEnableFrustumCullingAttr().Set(frustum_culling)

        self._tileset_path = tileset_path
        self._active = True
        self._start_time = time.time()

    def _tileset_loaded(self, _e: carb.events.IEvent):
        self._stop()
        duration = self._get_duration()
        self._update_duration_ui(duration)
        self._logger.warning("Tileset loaded in {} seconds".format(duration))

    def _get_duration(self) -> float:
        current_time = time.time()
        duration = current_time - self._start_time
        return duration

    def _update_duration_ui(self, duration: float):
        if self._performance_window is not None:
            self._performance_window.set_duration(duration)

    def _clear_scene(self):
        self._stop()
        self._update_duration_ui(0.0)

        if self._tileset_path is not None:
            _cesium_omniverse_interface.remove_tileset(self._tileset_path)

    def _on_stop(self, _e: carb.events.IEvent):
        self._stop()

    def _stop(self):
        self._active = False

        if self._tileset_loaded_subscription is not None:
            self._tileset_loaded_subscription.unsubscribe()
            self._tileset_loaded_subscription = None

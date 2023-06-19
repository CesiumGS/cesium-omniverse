from functools import partial
import asyncio
from typing import Optional
import logging
import carb.events
import omni.ext
import omni.ui as ui
import omni.kit.app as app
import omni.kit.ui
from .performance_window import CesiumPerformanceWindow
from cesium.omniverse.bindings import acquire_cesium_omniverse_interface, release_cesium_omniverse_interface
from cesium.omniverse.utils import wait_n_frames, dock_window_async


class CesiumPerformanceExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()

        self._logger = logging.getLogger(__name__)

        self._performance_window: Optional[CesiumPerformanceWindow] = None
        self._view_new_york_city_subscription: Optional[carb.events.ISubscription] = None
        self._view_grand_canyon_subscription: Optional[carb.events.ISubscription] = None
        self._view_tour_subscription: Optional[carb.events.ISubscription] = None

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

    def on_shutdown(self):
        self._destroy_performance_window()

        if self._view_new_york_city_subscription is not None:
            self._view_new_york_city_subscription.unsubscribe()
            self._view_new_york_city_subscription = None

        if self._view_grand_canyon_subscription is not None:
            self._view_grand_canyon_subscription.unsubscribe()
            self._view_grand_canyon_subscription = None

        if self._view_tour_subscription is not None:
            self._view_tour_subscription.unsubscribe()
            self._view_tour_subscription = None

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

    def _view_new_york_city(self, _: carb.events.IEvent):
        self._logger.warning("View NYC")

    def _view_grand_canyon(self, _: carb.events.IEvent):
        self._logger.warning("View Grand Canyon")

    def _view_tour(self, _: carb.events.IEvent):
        self._logger.warning("View Tour")

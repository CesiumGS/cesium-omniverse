from functools import partial
import asyncio
import time
from typing import Callable, List, Optional
import logging
import carb.events
import omni.ext
import omni.ui as ui
import omni.usd
import omni.kit.app as app
import omni.kit.ui
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, Sdf, UsdGeom
from .performance_window import CesiumPerformanceWindow
from .fps_sampler import FpsSampler
from cesium.omniverse.bindings import acquire_cesium_omniverse_interface, release_cesium_omniverse_interface
from cesium.omniverse.utils import wait_n_frames, dock_window_async
from cesium.usd.plugins.CesiumUsdSchemas import (
    Data as CesiumData,
    Georeference as CesiumGeoreference,
    Imagery as CesiumImagery,
    Tileset as CesiumTileset,
    Tokens as CesiumTokens,
)

ION_ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyZTA0MDlmYi01Y2RhLTQ0MjQtYjBlOS1kMmZhMzQ0OWRkNGYiLCJpZCI6MjU5LCJpYXQiOjE2ODU2MzExMTF9.y2CrqatkaHKHcj6NIDJ8ioll-tnOi-2CblnzI6iUays"  # noqa: E501
GOOGLE_3D_TILES_URL = "https://tile.googleapis.com/v1/3dtiles/root.json?key=AIzaSyC2PMYr_ZaMJT5DdZ8WJNYMwB0lDyvx5q8"

CESIUM_DATA_PRIM_PATH = "/Cesium"
CESIUM_GEOREFERENCE_PRIM_PATH = "/CesiumGeoreference"
CESIUM_CAMERA_PATH = "/Camera"


class CesiumPerformanceExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()

        self._logger = logging.getLogger(__name__)

        self._performance_window: Optional[CesiumPerformanceWindow] = None

        self._view_new_york_city_subscription: Optional[carb.events.ISubscription] = None
        self._view_paris_subscription: Optional[carb.events.ISubscription] = None
        self._view_grand_canyon_subscription: Optional[carb.events.ISubscription] = None
        self._view_tour_subscription: Optional[carb.events.ISubscription] = None
        self._view_new_york_city_google_subscription: Optional[carb.events.ISubscription] = None
        self._view_paris_google_subscription: Optional[carb.events.ISubscription] = None
        self._view_grand_canyon_google_subscription: Optional[carb.events.ISubscription] = None
        self._view_tour_google_subscription: Optional[carb.events.ISubscription] = None

        self._stop_subscription: Optional[carb.events.ISubscription] = None
        self._on_stage_subscription: Optional[carb.events.ISubscription] = None
        self._update_frame_subscription: Optional[carb.events.ISubscription] = None

        self._tileset_loaded_subscription: Optional[carb.events.ISubscription] = None

        self._camera_path: Optional[str] = None
        self._tileset_path: Optional[str] = None
        self._active: bool = False
        self._start_time: float = 0.0
        self._fps_sampler: FpsSampler = FpsSampler()

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

        view_paris_event = carb.events.type_from_string("cesium.performance.VIEW_PARIS")
        self._view_paris_subscription = bus.create_subscription_to_pop_by_type(view_paris_event, self._view_paris)

        view_grand_canyon_event = carb.events.type_from_string("cesium.performance.VIEW_GRAND_CANYON")
        self._view_grand_canyon_subscription = bus.create_subscription_to_pop_by_type(
            view_grand_canyon_event, self._view_grand_canyon
        )

        view_tour_event = carb.events.type_from_string("cesium.performance.VIEW_TOUR")
        self._view_tour_subscription = bus.create_subscription_to_pop_by_type(view_tour_event, self._view_tour)

        view_new_york_city_google_event = carb.events.type_from_string("cesium.performance.VIEW_NEW_YORK_CITY_GOOGLE")
        self._view_new_york_city_google_subscription = bus.create_subscription_to_pop_by_type(
            view_new_york_city_google_event, self._view_new_york_city_google
        )

        view_paris_google_event = carb.events.type_from_string("cesium.performance.VIEW_PARIS_GOOGLE")
        self._view_paris_google_subscription = bus.create_subscription_to_pop_by_type(
            view_paris_google_event, self._view_paris_google
        )

        view_grand_canyon_google_event = carb.events.type_from_string("cesium.performance.VIEW_GRAND_CANYON_GOOGLE")
        self._view_grand_canyon_google_subscription = bus.create_subscription_to_pop_by_type(
            view_grand_canyon_google_event, self._view_grand_canyon_google
        )

        view_tour_google_event = carb.events.type_from_string("cesium.performance.VIEW_TOUR_GOOGLE")
        self._view_tour_google_subscription = bus.create_subscription_to_pop_by_type(
            view_tour_google_event, self._view_tour_google
        )

        stop_event = carb.events.type_from_string("cesium.performance.STOP")
        self._stop_subscription = bus.create_subscription_to_pop_by_type(stop_event, self._on_stop)

        usd_context = omni.usd.get_context()
        if usd_context.get_stage_state() == omni.usd.StageState.OPENED:
            self._on_stage_opened()

        self._on_stage_subscription = usd_context.get_stage_event_stream().create_subscription_to_pop(
            self._on_stage_event, name="cesium.performance.ON_STAGE_EVENT"
        )

        update_stream = app.get_app().get_update_event_stream()
        self._update_frame_subscription = update_stream.create_subscription_to_pop(
            self._on_update_frame, name="cesium.performance.ON_UPDATE_FRAME"
        )

    def on_shutdown(self):
        self._clear_scene()

        if self._view_new_york_city_subscription is not None:
            self._view_new_york_city_subscription.unsubscribe()
            self._view_new_york_city_subscription = None

        if self._view_paris_subscription is not None:
            self._view_paris_subscription.unsubscribe()
            self._view_paris_subscription = None

        if self._view_grand_canyon_subscription is not None:
            self._view_grand_canyon_subscription.unsubscribe()
            self._view_grand_canyon_subscription = None

        if self._view_tour_subscription is not None:
            self._view_tour_subscription.unsubscribe()
            self._view_tour_subscription = None

        if self._view_new_york_city_google_subscription is not None:
            self._view_new_york_city_google_subscription.unsubscribe()
            self._view_new_york_city_google_subscription = None

        if self._view_paris_google_subscription is not None:
            self._view_paris_google_subscription.unsubscribe()
            self._view_paris_google_subscription = None

        if self._view_grand_canyon_google_subscription is not None:
            self._view_grand_canyon_google_subscription.unsubscribe()
            self._view_grand_canyon_google_subscription = None

        if self._view_tour_google_subscription is not None:
            self._view_tour_google_subscription.unsubscribe()
            self._view_tour_google_subscription = None

        if self._stop_subscription is not None:
            self._stop_subscription.unsubscribe()
            self._stop_subscription = None

        if self._on_stage_subscription is not None:
            self._on_stage_subscription.unsubscribe()
            self._on_stage_subscription = None

        if self._update_frame_subscription is not None:
            self._update_frame_subscription.unsubscribe()
            self._update_frame_subscription = None

        self._fps_sampler.destroy()

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
            self._update_fps_ui(self._fps_sampler.get_fps())

    def _on_stage_event(self, _e: carb.events.IEvent):
        usd_context = omni.usd.get_context()
        if usd_context.get_stage_state() == omni.usd.StageState.OPENED:
            self._on_stage_opened()

    def _on_stage_opened(self):
        self._camera_path = self._create_camera(CESIUM_CAMERA_PATH)

    @staticmethod
    def _create_tileset_ion(path: str, asset_id: int, access_token: str) -> str:
        stage = omni.usd.get_context().get_stage()
        tileset_path = omni.usd.get_stage_next_free_path(stage, path, False)
        tileset = CesiumTileset.Define(stage, tileset_path)
        assert tileset.GetPrim().IsValid()

        tileset.GetIonAssetIdAttr().Set(asset_id)
        tileset.GetIonAccessTokenAttr().Set(access_token)
        tileset.GetSourceTypeAttr().Set(CesiumTokens.ion)

        return tileset_path

    @staticmethod
    def _create_tileset_google() -> str:
        stage = omni.usd.get_context().get_stage()
        tileset_path = omni.usd.get_stage_next_free_path(stage, "/Google_3D_Tiles", False)
        tileset = CesiumTileset.Define(stage, tileset_path)

        tileset.GetUrlAttr().Set(GOOGLE_3D_TILES_URL)
        tileset.GetSourceTypeAttr().Set(CesiumTokens.url)

        return tileset_path

    @staticmethod
    def _create_imagery_ion(path: str, asset_id: int, access_token: str) -> str:
        stage = omni.usd.get_context().get_stage()
        imagery_path = omni.usd.get_stage_next_free_path(stage, path, False)
        imagery = CesiumImagery.Define(stage, imagery_path)
        assert imagery.GetPrim().IsValid()
        parent = imagery.GetPrim().GetParent()
        assert parent.IsA(CesiumTileset)

        imagery.GetIonAssetIdAttr().Set(asset_id)
        imagery.GetIonAccessTokenAttr().Set(access_token)

        return imagery_path

    @staticmethod
    def _create_camera(path: str) -> str:
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(path):
            return path

        camera = UsdGeom.Camera.Define(stage, path)
        assert camera.GetPrim().IsValid()

        camera.GetClippingRangeAttr().Set(Gf.Vec2f(1.0, 100000000.0))

        return path

    @staticmethod
    def _get_imagery_path(tileset_path: str, imagery_name: str) -> str:
        return Sdf.Path(tileset_path).AppendPath(imagery_name).pathString

    @staticmethod
    def _set_georeference(longitude: float, latitude: float, height: float):
        stage = omni.usd.get_context().get_stage()
        cesium_georeference = CesiumGeoreference.Get(stage, CESIUM_GEOREFERENCE_PRIM_PATH)
        assert cesium_georeference.GetPrim().IsValid()
        cesium_georeference.GetGeoreferenceOriginLongitudeAttr().Set(longitude)
        cesium_georeference.GetGeoreferenceOriginLatitudeAttr().Set(latitude)
        cesium_georeference.GetGeoreferenceOriginHeightAttr().Set(height)

    def _set_camera(
        self,
        translate: Gf.Vec3d,
        rotate: Gf.Vec3f,
        focal_length: float,
        horizontal_aperture: float,
        vertical_aperture: float,
    ):
        stage = omni.usd.get_context().get_stage()
        viewport = get_active_viewport()
        viewport.set_active_camera(self._camera_path)

        camera = UsdGeom.Camera.Get(stage, self._camera_path)
        camera.GetFocalLengthAttr().Set(focal_length)
        camera.GetHorizontalApertureAttr().Set(horizontal_aperture)
        camera.GetVerticalApertureAttr().Set(vertical_aperture)

        xform_common_api = UsdGeom.XformCommonAPI(camera.GetPrim())
        xform_common_api.SetTranslate(translate)
        xform_common_api.SetRotate(rotate, UsdGeom.XformCommonAPI.RotationOrderYXZ)

    @staticmethod
    def _get_tileset(path: str) -> CesiumTileset:
        stage = omni.usd.get_context().get_stage()
        tileset = CesiumTileset.Get(stage, path)
        assert tileset.GetPrim().IsValid()
        return tileset

    @staticmethod
    def _get_cesium_data() -> CesiumData:
        stage = omni.usd.get_context().get_stage()
        cesium_data = CesiumData.Get(stage, CESIUM_DATA_PRIM_PATH)
        assert cesium_data.GetPrim().IsValid()
        return cesium_data

    @staticmethod
    def _remove_prim(path: str):
        stage = omni.usd.get_context().get_stage()
        stage.RemovePrim(path)

    def _setup_location_new_york_city(self):
        self._set_georeference(-74.0060, 40.7128, 50.0)
        self._set_camera(
            Gf.Vec3d(-176516.8372437113, 33877.019622553846, 197777.19771945066),
            Gf.Vec3f(-7.9392824, -37.71652, -6.0970836),
            18.14756,
            20.955,
            15.2908,
        )

    def _setup_location_paris(self):
        self._set_georeference(2.3522, 48.8566, 100.0)
        self._set_camera(
            Gf.Vec3d(-285275.1368718885, 780.3607448845705, 35392.91845506678),
            Gf.Vec3f(0.46399376, 65.245544, -1.0061567),
            18.14756,
            20.955,
            15.2908,
        )

    def _setup_location_grand_canyon(self):
        self._set_georeference(-112.3535, 36.2679, 2100.0)
        self._set_camera(
            Gf.Vec3d(-339866.7567928189, 27967.440239271935, -59650.894693908194),
            Gf.Vec3f(5.532731, -129.35608, -6.704948),
            18.14756,
            20.955,
            15.2908,
        )

    def _view_new_york_city(self, _e: carb.events.IEvent):
        self._logger.warning("View New York City")
        self._clear_scene()
        self._setup_location_new_york_city()
        tileset_path = self._create_tileset_ion("/Cesium_World_Terrain", 1, ION_ACCESS_TOKEN)
        self._create_imagery_ion(
            self._get_imagery_path(tileset_path, "Bing_Maps_Aerial_Imagery"),
            2,
            ION_ACCESS_TOKEN,
        )
        self._load_tileset(tileset_path, self._tileset_loaded)

    def _view_paris(self, _e: carb.events.IEvent):
        self._logger.warning("View Paris")
        self._clear_scene()
        self._setup_location_paris()
        tileset_path = self._create_tileset_ion("/Cesium_World_Terrain", 1, ION_ACCESS_TOKEN)
        self._create_imagery_ion(
            self._get_imagery_path(tileset_path, "Bing_Maps_Aerial_Imagery"),
            2,
            ION_ACCESS_TOKEN,
        )
        self._load_tileset(tileset_path, self._tileset_loaded)

    def _view_grand_canyon(self, _e: carb.events.IEvent):
        self._logger.warning("View Grand Canyon")
        self._clear_scene()
        self._setup_location_grand_canyon()
        tileset_path = self._create_tileset_ion("/Cesium_World_Terrain", 1, ION_ACCESS_TOKEN)
        self._create_imagery_ion(
            self._get_imagery_path(tileset_path, "Bing_Maps_Aerial_Imagery"),
            2,
            ION_ACCESS_TOKEN,
        )
        self._load_tileset(tileset_path, self._tileset_loaded)

    def _view_tour(self, _e: carb.events.IEvent):
        self._logger.warning("View Tour")
        self._clear_scene()
        tileset_path = self._create_tileset_ion("/Cesium_World_Terrain", 1, ION_ACCESS_TOKEN)
        self._create_imagery_ion(
            self._get_imagery_path(tileset_path, "Bing_Maps_Aerial_Imagery"),
            2,
            ION_ACCESS_TOKEN,
        )

        def tour_stop_0():
            self._setup_location_new_york_city()

        def tour_stop_1():
            self._setup_location_paris()

        def tour_stop_2():
            self._setup_location_grand_canyon()

        tour = Tour(self, [tour_stop_0, tour_stop_1, tour_stop_2], self._tileset_loaded)

        self._load_tileset(tileset_path, tour.tour_stop_loaded)

    def _view_new_york_city_google(self, _e: carb.events.IEvent):
        self._logger.warning("View New York City Google")
        self._clear_scene()
        self._setup_location_new_york_city()
        tileset_path = self._create_tileset_google()
        self._load_tileset(tileset_path, self._tileset_loaded)

    def _view_paris_google(self, _e: carb.events.IEvent):
        self._logger.warning("View Paris Google")
        self._clear_scene()
        self._setup_location_paris()
        tileset_path = self._create_tileset_google()
        self._load_tileset(tileset_path, self._tileset_loaded)

    def _view_grand_canyon_google(self, _e: carb.events.IEvent):
        self._logger.warning("View Grand Canyon Google")
        self._clear_scene()
        self._setup_location_grand_canyon()
        tileset_path = self._create_tileset_google()
        self._load_tileset(tileset_path, self._tileset_loaded)

    def _view_tour_google(self, _e: carb.events.IEvent):
        self._logger.warning("View Tour Google")
        self._clear_scene()
        tileset_path = self._create_tileset_google()

        def tour_stop_0():
            self._setup_location_new_york_city()

        def tour_stop_1():
            self._setup_location_paris()

        def tour_stop_2():
            self._setup_location_grand_canyon()

        tour = Tour(self, [tour_stop_0, tour_stop_1, tour_stop_2], self._tileset_loaded)

        self._load_tileset(tileset_path, tour.tour_stop_loaded)

    def _load_tileset(self, tileset_path: str, tileset_loaded: Callable):
        tileset = self._get_tileset(tileset_path)
        cesium_data = self._get_cesium_data()

        assert self._performance_window is not None

        bus = app.get_app().get_message_bus_event_stream()
        tileset_loaded_event = carb.events.type_from_string("cesium.omniverse.TILESET_LOADED")
        self._tileset_loaded_subscription = bus.create_subscription_to_pop_by_type(
            tileset_loaded_event, tileset_loaded
        )

        random_colors = self._performance_window.get_random_colors()
        forbid_holes = self._performance_window.get_forbid_holes()
        frustum_culling = self._performance_window.get_frustum_culling()

        cesium_data.GetDebugRandomColorsAttr().Set(random_colors)
        tileset.GetForbidHolesAttr().Set(forbid_holes)
        tileset.GetEnableFrustumCullingAttr().Set(frustum_culling)

        self._tileset_path = tileset_path
        self._active = True
        self._start_time = time.time()
        self._fps_sampler.start()

    def _tileset_loaded(self, _e: carb.events.IEvent):
        self._stop()
        duration = self._get_duration()
        self._update_duration_ui(duration)
        self._update_fps_mean_ui(self._fps_sampler.get_mean())
        self._update_fps_median_ui(self._fps_sampler.get_median())
        self._update_fps_low_ui(self._fps_sampler.get_low())
        self._update_fps_high_ui(self._fps_sampler.get_high())
        self._logger.warning("Loaded in {} seconds".format(duration))

    def _get_duration(self) -> float:
        current_time = time.time()
        duration = current_time - self._start_time
        return duration

    def _update_duration_ui(self, value: float):
        if self._performance_window is not None:
            self._performance_window.set_duration(value)

    def _update_fps_ui(self, value: float):
        if self._performance_window is not None:
            self._performance_window.set_fps(value)

    def _update_fps_mean_ui(self, value: float):
        if self._performance_window is not None:
            self._performance_window.set_fps_mean(value)

    def _update_fps_median_ui(self, value: float):
        if self._performance_window is not None:
            self._performance_window.set_fps_median(value)

    def _update_fps_low_ui(self, value: float):
        if self._performance_window is not None:
            self._performance_window.set_fps_low(value)

    def _update_fps_high_ui(self, value: float):
        if self._performance_window is not None:
            self._performance_window.set_fps_high(value)

    def _clear_scene(self):
        self._stop()
        self._update_duration_ui(0.0)
        self._update_fps_ui(0.0)
        self._update_fps_mean_ui(0.0)
        self._update_fps_median_ui(0.0)
        self._update_fps_low_ui(0.0)
        self._update_fps_high_ui(0.0)

        if self._tileset_path is not None:
            self._remove_prim(self._tileset_path)

    def _on_stop(self, _e: carb.events.IEvent):
        self._stop()

    def _stop(self):
        self._active = False

        self._fps_sampler.stop()

        if self._tileset_loaded_subscription is not None:
            self._tileset_loaded_subscription.unsubscribe()
            self._tileset_loaded_subscription = None


class Tour:
    def __init__(self, ext: CesiumPerformanceExtension, tour_stops: List[Callable], tour_complete: Callable):
        self._ext: CesiumPerformanceExtension = ext
        self._tour_stops: List[Callable] = tour_stops
        self._tour_complete: Callable = tour_complete
        self._current_stop: int = 0
        self._duration: float = 0.0

        assert len(tour_stops) > 0
        tour_stops[0]()

    def tour_stop_loaded(self, _e: carb.events.IEvent):
        duration = self._ext._get_duration()
        current_duration = duration - self._duration
        self._duration = duration

        self._ext._logger.warning("Tour stop {} loaded in {} seconds".format(self._current_stop, current_duration))

        if self._current_stop == len(self._tour_stops) - 1:
            self._tour_complete(_e)
        else:
            self._current_stop += 1
            self._tour_stops[self._current_stop]()

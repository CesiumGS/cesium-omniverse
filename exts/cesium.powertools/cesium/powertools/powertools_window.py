import logging
import omni.ui as ui
from typing import Callable, Optional, List
from cesium.omniverse.ui import CesiumOmniverseDebugWindow
from .georefhelper.georef_helper_window import CesiumGeorefHelperWindow
from .utils import extend_far_plane, save_carb_settings, save_fabric_stage, set_sunstudy_from_georef
import os
from functools import partial

powertools_extension_location = os.path.join(os.path.dirname(__file__), "../../")


class PowertoolsAction:
    def __init__(self, title: str, action: Callable):
        self._title = title
        self._action = action
        self._button: Optional[ui.Button] = None

    def destroy(self):
        if self._button is not None:
            self._button.destroy()
        self._button = None

    def button(self):
        if self._button is None:
            self._button = ui.Button(self._title, height=0, clicked_fn=self._action)

        return self._button


class CesiumPowertoolsWindow(ui.Window):
    WINDOW_NAME = "Cesium Power Tools"
    MENU_PATH = f"Window/Cesium/{WINDOW_NAME}"

    def __init__(self, **kwargs):
        super().__init__(CesiumPowertoolsWindow.WINDOW_NAME, **kwargs)

        self._logger = logging.getLogger(__name__)

        # You do not necessarily need to create an action function in this window class. If you have a function
        #  in another window or class, you can absolutely call that instead from here.
        self._actions: List[PowertoolsAction] = [
            PowertoolsAction("Open Cesium Debugging Window", CesiumOmniverseDebugWindow.show_window),
            PowertoolsAction("Open Cesium Georeference Helper Window", CesiumGeorefHelperWindow.create_window),
            PowertoolsAction("Extend Far Plane", extend_far_plane),
            PowertoolsAction("Save Carb Settings", partial(save_carb_settings, powertools_extension_location)),
            PowertoolsAction("Save Fabric Stage", partial(save_fabric_stage, powertools_extension_location)),
            PowertoolsAction("Set Sun Study from Georef", set_sunstudy_from_georef),
        ]

        self.frame.set_build_fn(self._build_fn)

    def destroy(self) -> None:
        for action in self._actions:
            action.destroy()
        self._actions.clear()

        super().destroy()

    def _build_fn(self):
        with ui.VStack(spacing=4):
            for action in self._actions:
                action.button()

import logging
import carb.events
import omni.kit.app as app
import omni.ui as ui
from omni.kit.viewport.utility import get_active_viewport_window
from typing import List, Optional, Tuple
from ..bindings import ICesiumOmniverseInterface
from .credits_parser import CesiumCreditsParser
from .credits_window import CesiumOmniverseCreditsWindow


class CesiumCreditsViewportFrame:
    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface):
        self._logger = logging.getLogger(__name__)

        self._cesium_omniverse_interface = cesium_omniverse_interface

        viewport_window = get_active_viewport_window()
        self._credits_viewport_frame = viewport_window.get_frame("cesium.omniverse.viewport.ION_CREDITS")

        self._credits_window: Optional[CesiumOmniverseCreditsWindow] = None
        self._data_attribution_button: Optional[ui.Button] = None

        self._subscriptions: List[carb.events.ISubscription] = []
        self._setup_subscriptions()

        self._credits: List[Tuple[str, bool]] = []

        self._build_fn()

    def __del__(self):
        self.destroy()

    def destroy(self):
        for subscription in self._subscriptions:
            subscription.unsubscribe()
        self._subscriptions.clear()

        if self._credits_window is not None:
            self._credits_window.destroy()
            self._credits_window = None

    def _setup_subscriptions(self):
        update_stream = app.get_app().get_update_event_stream()
        self._subscriptions.append(
            update_stream.create_subscription_to_pop(
                self._on_update_frame, name="cesium.omniverse.viewport.ON_UPDATE_FRAME"
            )
        )

    def _on_update_frame(self, _e: carb.events.IEvent):
        if self._data_attribution_button is None:
            return

        credits_available = self._cesium_omniverse_interface.credits_available()

        if credits_available != self._data_attribution_button.visible:
            if credits_available:
                self._logger.info("Show Data Attribution")
            else:
                self._logger.info("Hide Data Attribution")
            self._data_attribution_button.visible = credits_available

        if self._data_attribution_button.visible:
            new_credits = self._cesium_omniverse_interface.get_credits()
            if new_credits is not None and len(self._credits) != len(new_credits):
                self._credits.clear()
                self._credits.extend(new_credits)
                self._build_fn()
        else:
            self._credits.clear()
            self._build_fn()

    def _on_data_attribution_button_clicked(self):
        self._credits_window = CesiumOmniverseCreditsWindow(self._cesium_omniverse_interface)

    def _build_fn(self):
        with self._credits_viewport_frame:
            with ui.VStack():
                ui.Spacer()
                with ui.HStack(height=0):
                    ui.Spacer()

                    CesiumCreditsParser(self._credits, should_show_on_screen=True)

                    self._data_attribution_button = ui.Button(
                        "Data Attribution",
                        visible=False,
                        width=0,
                        height=0,
                        clicked_fn=self._on_data_attribution_button_clicked,
                    )

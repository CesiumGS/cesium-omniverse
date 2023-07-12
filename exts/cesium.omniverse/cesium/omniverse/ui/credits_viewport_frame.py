import logging
import carb.events
import omni.kit.app as app
import omni.ui as ui
from typing import List, Optional, Tuple
from ..bindings import ICesiumOmniverseInterface
from .credits_parser import CesiumCreditsParser
from .credits_window import CesiumOmniverseCreditsWindow
import json
from .events import EVENT_CREDITS_CHANGED


class CesiumCreditsViewportFrame:
    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, instance):
        self._logger = logging.getLogger(__name__)

        self._cesium_omniverse_interface = cesium_omniverse_interface

        self._credits_viewport_frame = instance.get_frame("cesium.omniverse.viewport.ION_CREDITS")

        self._credits_window: Optional[CesiumOmniverseCreditsWindow] = None
        self._data_attribution_button: Optional[ui.Button] = None

        self._on_credits_changed_event = EVENT_CREDITS_CHANGED
        self._subscriptions: List[carb.events.ISubscription] = []
        self._setup_subscriptions()

        self._credits: List[Tuple[str, bool]] = []
        self._new_credits: List[Tuple[str, bool]] = []

        self._build_fn()

    def getFrame(self):
        return self._credits_viewport_frame

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
        message_bus = app.get_app().get_message_bus_event_stream()
        self._subscriptions.append(
            message_bus.create_subscription_to_pop_by_type(EVENT_CREDITS_CHANGED, self._on_credits_changed)
        )

    def _on_update_frame(self, _e: carb.events.IEvent):
        if self._data_attribution_button is None:
            return

        if self._new_credits != self._credits:
            self._credits.clear()
            self._credits.extend(self._new_credits)
            self._build_fn()

        has_offscreen_credits = False
        for _, show_on_screen in self._new_credits:
            if not show_on_screen:
                has_offscreen_credits = True

        if has_offscreen_credits != self._data_attribution_button.visible:
            if has_offscreen_credits:
                self._logger.info("Show Data Attribution")
            else:
                self._logger.info("Hide Data Attribution")
            self._data_attribution_button.visible = has_offscreen_credits

    def _on_data_attribution_button_clicked(self):
        self._credits_window = CesiumOmniverseCreditsWindow(self._cesium_omniverse_interface, self._credits)

    def _build_fn(self):
        with self._credits_viewport_frame:
            with ui.VStack():
                ui.Spacer()
                with ui.HStack(height=0):
                    # Prevent credits from overlapping the axis display
                    ui.Spacer(width=100)

                    with ui.HStack(height=0, spacing=4):
                        CesiumCreditsParser(
                            self._credits,
                            should_show_on_screen=True,
                            combine_labels=True,
                            label_alignment=ui.Alignment.RIGHT,
                        )

                    # VStack + Spacer pushes our content to the bottom of the Stack to account for varying heights
                    with ui.VStack(spacing=0, width=0):
                        ui.Spacer()
                        self._data_attribution_button = ui.Button(
                            "Data Attribution",
                            visible=False,
                            width=0,
                            height=0,
                            clicked_fn=self._on_data_attribution_button_clicked,
                        )

    def _on_credits_changed(self, _e: carb.events.IEvent):
        credits_json = _e.payload["credits"]
        credits = json.loads(credits_json)
        if credits is not None:
            self._new_credits = credits

from .credits_parser import CesiumCreditsParser, ParsedCredit
from typing import List, Optional, Tuple
import logging
import carb.events
import omni.kit.app as app
from ..bindings import ICesiumOmniverseInterface
import omni.ui as ui
import omni.kit.app
import json
from carb.events import IEventStream
from .events import EVENT_CREDITS_CHANGED


class CreditsViewportController:
    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface):
        self._cesium_omniverse_interface = cesium_omniverse_interface
        self._logger: Optional[logging.Logger] = logging.getLogger(__name__)
        self._parsed_credits: List[ParsedCredit] = []
        self._credits: List[Tuple[str, bool]] = []
        self._subscriptions: List[carb.events.ISubscription] = []

        self._setup_update_subscription()
        self._message_bus: IEventStream = omni.kit.app.get_app().get_message_bus_event_stream()
        self._EVENT_CREDITS_CHANGED: int = EVENT_CREDITS_CHANGED

    def __del__(self):
        self.destroy()

    def destroy(self):
        for subscription in self._subscriptions:
            subscription.unsubscribe()
        self._subscriptions.clear()

    def _setup_update_subscription(self):
        update_stream = app.get_app().get_update_event_stream()
        self._subscriptions.append(
            update_stream.create_subscription_to_pop(
                self._on_update_frame, name="cesium.omniverse.viewport.ON_UPDATE_FRAME"
            )
        )

    def _on_update_frame(self, _e: carb.events.IEvent):
        if self._cesium_omniverse_interface is None:
            return

        new_credits = self._cesium_omniverse_interface.get_credits()
        # cheap test
        if new_credits != self._credits:
            self._credits.clear()
            self._credits.extend(new_credits)

            # deep test
            credits_parser = CesiumCreditsParser(
                new_credits,
                should_show_on_screen=True,
                combine_labels=True,
                label_alignment=ui.Alignment.RIGHT,
            )
            new_parsed_credits = credits_parser._parse_credits(new_credits, True, False)
            if new_parsed_credits != self._parsed_credits:
                self._parsed_credits = new_parsed_credits
                self.broadcast_credits()

        self._cesium_omniverse_interface.credits_start_next_frame()

    def broadcast_credits(self):
        my_payload = json.dumps(self._credits)
        self._message_bus.push(self._EVENT_CREDITS_CHANGED, payload={"credits": my_payload})

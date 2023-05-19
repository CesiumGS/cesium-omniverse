from .credits_parser import CesiumCreditsParser, ParsedCredit
from typing import List, Optional, Tuple
import logging
import carb.events
import omni.kit.app as app
from ..bindings import ICesiumOmniverseInterface
import omni.ui as ui


class CreditsViewportController:
    _instance = None
    _cesium_omniverse_interface = None
    _some_string = None
    _logger: Optional[logging.Logger] = None
    _parsed_credits: List[ParsedCredit] = []
    _credits: List[Tuple[str, bool]] = []
    _event_handlers = []
    _subscriptions: List[carb.events.ISubscription] = []
    _initted: bool = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not CreditsViewportController._initted:
            CreditsViewportController._logger = logging.getLogger(__name__)
            self._setup_subscriptions()
            CreditsViewportController._initted = True

    def __del__(self):
        if CreditsViewportController._instance is not None:
            self.destroy()

    def destroy(self):
        for subscription in CreditsViewportController._subscriptions:
            subscription.unsubscribe()
        CreditsViewportController._subscriptions.clear()

    def _setup_subscriptions(self):
        update_stream = app.get_app().get_update_event_stream()
        CreditsViewportController._subscriptions.append(
            update_stream.create_subscription_to_pop(
                self._on_update_frame, name="cesium.omniverse.viewport.ON_UPDATE_FRAME"
            )
        )

    def _on_update_frame(self, _e: carb.events.IEvent):
        if CreditsViewportController._cesium_omniverse_interface is None:
            return

        new_credits = CreditsViewportController._cesium_omniverse_interface.get_credits()
        # cheap test
        if (new_credits != CreditsViewportController._credits):
            CreditsViewportController._logger.info("CreditsViewportController credits have changed")
            CreditsViewportController._credits.clear()
            CreditsViewportController._credits.extend(new_credits)

            # deep test
            credits_parser = CesiumCreditsParser(
                                new_credits,
                                should_show_on_screen=True,
                                combine_labels=True,
                                label_alignment=ui.Alignment.RIGHT,
                            )
            new_parsed_credits = credits_parser._parse_credits(new_credits, True, False)
            if new_parsed_credits != CreditsViewportController._parsed_credits:
                CreditsViewportController._logger.info("CreditsViewportController: parsed credits changed")
                CreditsViewportController.send_event(self, new_credits)
                CreditsViewportController._parsed_credits = new_parsed_credits
        CreditsViewportController._cesium_omniverse_interface.credits_start_next_frame()

    def start(self, cesium_omniverse_interface: ICesiumOmniverseInterface):
        CreditsViewportController._cesium_omniverse_interface = cesium_omniverse_interface
        CreditsViewportController._logger.info("CreditsViewportController in start")

    def register_handler(self, handler):
        CreditsViewportController._event_handlers.append(handler)

    def send_event(self, event):
        for handler in CreditsViewportController._event_handlers:
            handler.handle_event(event)

    def clear_handlers(self):
        CreditsViewportController._event_handlers = []

    def get_current_credits(self):
        return CreditsViewportController._credits

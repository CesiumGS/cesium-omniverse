from .credits_parser import CesiumCreditsParser, ParsedCredit
from typing import List, Optional, Tuple
import logging
import carb.events
import omni.kit.app as app
from ..bindings import ICesiumOmniverseInterface
import omni.ui as ui


class CreditsController:
    _instance = None
    _cesium_omniverse_interface = None
    _some_string = None
    _logger: Optional[logging.Logger] = None
    _parsed_credits: List[ParsedCredit] = []
    _credits: List[Tuple[str, bool]] = []
    _event_handlers = []

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
            print("CreditsController __new__ run.")
        return cls._instance

    def __init__(self):
        print("CreditsController __init__ run.")
        CreditsController._logger = logging.getLogger(__name__)
        self._subscriptions: List[carb.events.ISubscription] = []
        self._setup_subscriptions()

    def __del__(self):
        if CreditsController._instance is not None:
            self.destroy()

    def destroy(self):
        for subscription in self._subscriptions:
            subscription.unsubscribe()
        self._subscriptions.clear()

    def _setup_subscriptions(self):
        update_stream = app.get_app().get_update_event_stream()
        self._subscriptions.append(
            update_stream.create_subscription_to_pop(
                self._on_update_frame, name="cesium.omniverse.viewport.ON_UPDATE_FRAME"
            )
        )

    def _on_update_frame(self, _e: carb.events.IEvent):
        if CreditsController._cesium_omniverse_interface is None:
            return

        new_credits = CreditsController._cesium_omniverse_interface.get_credits()
        # cheap test
        if (new_credits != CreditsController._credits):
            CreditsController._logger.info("CreditsController credits have changed")
            CreditsController._credits.clear()
            CreditsController._credits.extend(new_credits)

            # deep test
            credits_parser = CesiumCreditsParser(
                                new_credits,
                                should_show_on_screen=True,
                                combine_labels=True,
                                label_alignment=ui.Alignment.RIGHT,
                            )
            new_parsed_credits = credits_parser._parse_credits(new_credits, True, False)
            if new_parsed_credits != CreditsController._parsed_credits:
                CreditsController._logger.info("CreditsController: parsed credits changed")
                CreditsController.send_event(self, new_credits)
                CreditsController._parsed_credits = new_parsed_credits
        CreditsController._cesium_omniverse_interface.credits_start_next_frame()

    def start(self, cesium_omniverse_interface: ICesiumOmniverseInterface):
        CreditsController._cesium_omniverse_interface = cesium_omniverse_interface
        CreditsController._logger.info("CreditsController in start")

    def register_handler(self, handler):
        CreditsController._event_handlers.append(handler)

    def send_event(self, event):
        for handler in CreditsController._event_handlers:
            handler.handle_event(event)

    def clear_handlers(self):
        CreditsController._event_handlers = []

    def get_current_credits(self):
        return CreditsController._credits

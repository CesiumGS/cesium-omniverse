from .credits_parser import CesiumCreditsParser
from typing import List, Optional
import logging
import carb.events
import omni.kit.app as app
from ..bindings import ICesiumOmniverseInterface


class CreditsController:
    _instance = None
    _cesium_omniverse_interface = None
    _some_string = None
    _logger: Optional[logging.Logger] = None

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
        # if _cesium_omniverse_interface is None:
        #     return

        CreditsController._logger.info("CreditsController is updating the frame")
        new_credits = CreditsController._cesium_omniverse_interface.get_credits()
        # if new_credits != self._credits:
        #     self._credits.clear()
        #     self._credits.extend(new_credits)
        #     self._logger.info("CreditViewportFrame: credits changed, triggering CreditsViewportFrames setup")
        #     self._setup_credits_viewport_frames()
        #     self._credits = new_credits
        new_credits_len = len(new_credits)
        CreditsController._logger.info(f"CreditsController credits is length {new_credits_len}")
        CreditsController._cesium_omniverse_interface.credits_start_next_frame()

    def start(self, cesium_omniverse_interface: ICesiumOmniverseInterface):
        CreditsController._cesium_omniverse_interface = cesium_omniverse_interface
        CreditsController._logger.info("CreditsController in start")

from .credits_parser import CesiumCreditsParser
from typing import List
import logging
import carb.events
import omni.kit.app as app


class CreditsController:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
            print("CreditsController __new__ run.")
        return cls._instance

    def __init__(self):
        print("CreditsController __init__ run.")
        self._logger = logging.getLogger(__name__)
        self._subscriptions: List[carb.events.ISubscription] = []
        self._setup_subscriptions()

    def __del__(self):
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
        self._logger.info("CreditsController is updating the frame")

    def start(self):
        pass

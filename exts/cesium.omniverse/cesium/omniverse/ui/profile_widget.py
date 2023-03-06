import logging
import carb.events
import omni.kit.app as app
import omni.ui as ui
import webbrowser
from typing import List, Optional
from ..bindings import ICesiumOmniverseInterface

LOADING_PROFILE_MESSAGE = "Loading user information..."
CONNECTED_TO_MESSAGE_BASE = "Connected to Cesium ion as"


class CesiumOmniverseProfileWidget(ui.Frame):
    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, **kwargs):
        self._logger = logging.getLogger(__name__)
        self._cesium_omniverse_interface = cesium_omniverse_interface

        self._profile_id: Optional[int] = None
        self._button_enabled = False
        self._message = ""

        self._subscriptions: List[carb.events.ISubscription] = []
        self._setup_subscriptions()

        super().__init__(build_fn=self._build_ui, **kwargs)

    def destroy(self) -> None:
        for subscription in self._subscriptions:
            subscription.unsubscribe()
        self._subscriptions.clear()

    def _setup_subscriptions(self):
        update_stream = app.get_app().get_update_event_stream()
        self._subscriptions.append(
            update_stream.create_subscription_to_pop(self._on_update_frame, name="on_update_frame")
        )

    def _on_update_frame(self, _e: carb.events.IEvent):
        session = self._cesium_omniverse_interface.get_session()
        if session is not None:
            if session.is_profile_loaded():
                profile = session.get_profile()
                if self._profile_id != profile.id:
                    self._profile_id = profile.id
                    self._button_enabled = True
                    self._message = f"{CONNECTED_TO_MESSAGE_BASE} {profile.username}"
                    self.rebuild()
            elif session.is_loading_profile():
                self.visible = True
                self._button_enabled = False
                self._message = LOADING_PROFILE_MESSAGE
                self.rebuild()
            elif session.is_connected():
                session.refresh_profile()
            else:
                self._profile_id = None
                self.visible = False

    def _on_profile_button_clicked(self) -> None:
        if self.visible:
            # We just open a link to ion directly.
            # They may have to sign in if they aren't signed in on their browser already.
            webbrowser.open("https://cesium.com/ion")

    def _build_ui(self):
        with self:
            ui.Button(self._message, clicked_fn=self._on_profile_button_clicked, enabled=self._button_enabled)

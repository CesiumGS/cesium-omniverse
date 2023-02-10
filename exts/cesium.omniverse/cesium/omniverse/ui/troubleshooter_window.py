import logging
import carb.events
import omni.kit.app as app
import omni.ui as ui
import omni.usd
import webbrowser
from typing import List, Optional
from ..bindings import ICesiumOmniverseInterface, Token
from .styles import CesiumOmniverseUiStyles


class CesiumTroubleshooterWindow(ui.Window):
    WINDOW_NAME = "Token Troubleshooting"

    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, **kwargs):
        super().__init__(CesiumTroubleshooterWindow.WINDOW_NAME, **kwargs)

        self._cesium_omniverse_interface = cesium_omniverse_interface
        self._logger = logging.getLogger(__name__)

        self.height = 400
        self.width = 600

        self.padding_x = 12
        self.padding_y = 12

        self._subscriptions: List[carb.events.ISubscription] = []
        self._setup_subscriptions()

        self._token_details_event_type = carb.events.type_from_string("cesium.omniverse.TOKEN_DETAILS_READY")
        self._asset_details_event_type = carb.events.type_from_string("cesium.omniverse.ASSET_DETAILS_READY")

        self.frame.set_build_fn(self._build_ui)

    def __del__(self):
        self.destroy()

    def destroy(self):
        for subscription in self._subscriptions:
            subscription.unsubscribe()
        self._subscriptions = None

    def _setup_subscriptions(self):
        bus = app.get_app().get_message_bus_event_stream()

        self._subscriptions.append(
            bus.create_subscription_to_pop_by_type(self._token_details_event_type, self._on_token_details_ready,
                                                   name="cesium.omniverse.TOKEN_DETAILS_READY")
        )

        self._subscriptions.append(
            bus.create_subscription_to_pop_by_type(self._asset_details_event_type, self._on_asset_details_ready,
                                                   name="cesium.omniverse.ASSET_DETAILS_READY")
        )

    def _on_token_details_ready(self, _e: carb.events.IEvent):
        pass

    def _on_asset_details_ready(self, _e: carb.events.IEvent):
        pass

    def _on_open_ion_button_clicked(self):
        webbrowser.open("https://ion.cesium.com")

    def _build_ui(self):
        with ui.HStack():
            with ui.VStack():
                ui.Label("Stage Default Access Token")
                # TODO: Is Valid Token Check
                # TODO: Allows Access to this asset Check
                # TODO: Is associated to your account check
            with ui.VStack():
                ui.Label("Asset")
                # TODO: Asset ID exists in your user account check
            ui.Button("Open Cesium ion on the Web", alignment=ui.Alignment.CENTER, height=36,
                      style=CesiumOmniverseUiStyles.blue_button_style,
                      clicked_fn=self._on_open_ion_button_clicked)

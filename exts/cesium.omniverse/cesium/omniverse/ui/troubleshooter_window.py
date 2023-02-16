import logging
import carb.events
import omni.kit.app as app
import omni.ui as ui
import webbrowser
from typing import List, Optional
from ..bindings import ICesiumOmniverseInterface
from .pass_fail_widget import CesiumPassFailWidget
from .styles import CesiumOmniverseUiStyles


class CesiumTroubleshooterWindow(ui.Window):
    WINDOW_NAME = "Token Troubleshooting"

    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, tileset_id: int, raster_overlay_id: int,
                 message: str, **kwargs):
        super().__init__(CesiumTroubleshooterWindow.WINDOW_NAME, **kwargs)

        self._cesium_omniverse_interface = cesium_omniverse_interface
        self._logger = logging.getLogger(__name__)

        self._tileset_id = tileset_id
        self._raster_overlay_id = raster_overlay_id
        self._message = message

        self.height = 400
        self.width = 600

        self.padding_x = 12
        self.padding_y = 12

        self._token_details_event_type = carb.events.type_from_string("cesium.omniverse.TOKEN_DETAILS_READY")
        self._asset_details_event_type = carb.events.type_from_string("cesium.omniverse.ASSET_DETAILS_READY")

        self._valid_token_widget: Optional[CesiumPassFailWidget] = None
        self._token_has_access_widget: Optional[CesiumPassFailWidget] = None
        self._token_associated_to_account_widget: Optional[CesiumPassFailWidget] = None
        self._asset_on_account_widget: Optional[CesiumPassFailWidget] = None

        self._subscriptions: List[carb.events.ISubscription] = []
        self._setup_subscriptions()

        if raster_overlay_id > 0:
            self._cesium_omniverse_interface.update_troubleshooting_details(tileset_id, raster_overlay_id,
                                                                            self._token_details_event_type,
                                                                            self._asset_details_event_type)
        else:
            self._cesium_omniverse_interface.update_troubleshooting_details(tileset_id, self._token_details_event_type,
                                                                            self._asset_details_event_type)

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
        token_details = self._cesium_omniverse_interface.get_token_troubleshooting_details()

        if self._valid_token_widget is not None:
            self._valid_token_widget.passed = token_details.is_valid

        if self._token_has_access_widget is not None:
            self._token_has_access_widget.passed = token_details.allows_access_to_asset

        if self._token_associated_to_account_widget is not None:
            self._token_associated_to_account_widget.passed = token_details.associated_with_user_account

    def _on_asset_details_ready(self, _e: carb.events.IEvent):
        asset_details = self._cesium_omniverse_interface.get_asset_troubleshooting_details()

        if self._asset_on_account_widget is not None:
            self._asset_on_account_widget.passed = asset_details.asset_exists_in_user_account

    @staticmethod
    def _on_open_ion_button_clicked():
        webbrowser.open("https://ion.cesium.com")

    def _build_ui(self):
        with ui.VStack(spacing=10):
            ui.Label(self._message)
            with ui.HStack(spacing=5):
                with ui.VStack(spacing=5):
                    ui.Label("Stage Default Access Token", height=16,
                             style=CesiumOmniverseUiStyles.troubleshooter_header_style)
                    with ui.HStack(height=16, spacing=10):
                        self._valid_token_widget = CesiumPassFailWidget()
                        ui.Label("Is a valid Cesium ion Token")
                    with ui.HStack(height=16, spacing=10):
                        self._token_has_access_widget = CesiumPassFailWidget()
                        ui.Label("Allows access to this asset")
                    with ui.HStack(height=16, spacing=10):
                        self._token_associated_to_account_widget = CesiumPassFailWidget()
                        ui.Label("Is associated with your user account")
                with ui.VStack():
                    ui.Label("Asset", height=16, style=CesiumOmniverseUiStyles.troubleshooter_header_style)
                    with ui.HStack(height=16, spacing=10):
                        self._asset_on_account_widget = CesiumPassFailWidget()
                        ui.Label("Asset ID exists in your user account")
            ui.Spacer()
            ui.Button("Open Cesium ion on the Web", alignment=ui.Alignment.CENTER, height=36,
                      style=CesiumOmniverseUiStyles.blue_button_style,
                      clicked_fn=self._on_open_ion_button_clicked)

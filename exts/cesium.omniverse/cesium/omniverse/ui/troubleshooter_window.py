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
    WINDOW_BASE_NAME = "Token Troubleshooting"

    def __init__(
        self,
        cesium_omniverse_interface: ICesiumOmniverseInterface,
        name: str,
        tileset_path: str,
        tileset_ion_asset_id: int,
        raster_overlay_ion_asset_id: int,
        message: str,
        **kwargs,
    ):
        window_name = f"{CesiumTroubleshooterWindow.WINDOW_BASE_NAME} - {name}"

        super().__init__(window_name, **kwargs)

        self._cesium_omniverse_interface = cesium_omniverse_interface
        self._logger = logging.getLogger(__name__)

        self._name = name
        self._tileset_path = tileset_path
        self._tileset_ion_asset_id = tileset_ion_asset_id
        self.raster_overlay_ion_asset_id = raster_overlay_ion_asset_id

        ion_id = raster_overlay_ion_asset_id if raster_overlay_ion_asset_id > 0 else tileset_ion_asset_id
        self._message = (
            f"{name} tried to access Cesium ion for asset id {ion_id}, but it didn't work, probably "
            + "due to a problem with the access token. This panel will help you fix it!"
        )

        self.height = 400
        self.width = 700

        self.padding_x = 12
        self.padding_y = 12

        self._token_details_event_type = carb.events.type_from_string("cesium.omniverse.TOKEN_DETAILS_READY")
        self._asset_details_event_type = carb.events.type_from_string("cesium.omniverse.ASSET_DETAILS_READY")

        self._default_token_stack: Optional[ui.VStack] = None
        self._default_token_is_valid_widget: Optional[CesiumPassFailWidget] = None
        self._default_token_has_access_widget: Optional[CesiumPassFailWidget] = None
        self._default_token_associated_to_account_widget: Optional[CesiumPassFailWidget] = None
        self._asset_token_stack: Optional[ui.VStack] = None
        self._asset_token_is_valid_widget: Optional[CesiumPassFailWidget] = None
        self._asset_token_has_access_widget: Optional[CesiumPassFailWidget] = None
        self._asset_token_associated_to_account_widget: Optional[CesiumPassFailWidget] = None
        self._asset_on_account_widget: Optional[CesiumPassFailWidget] = None

        self._subscriptions: List[carb.events.ISubscription] = []
        self._setup_subscriptions()

        if raster_overlay_ion_asset_id > 0:
            self._cesium_omniverse_interface.update_troubleshooting_details(
                tileset_path,
                tileset_ion_asset_id,
                raster_overlay_ion_asset_id,
                self._token_details_event_type,
                self._asset_details_event_type,
            )
        else:
            self._cesium_omniverse_interface.update_troubleshooting_details(
                tileset_path, tileset_ion_asset_id, self._token_details_event_type, self._asset_details_event_type
            )

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
            bus.create_subscription_to_pop_by_type(
                self._token_details_event_type,
                self._on_token_details_ready,
                name="cesium.omniverse.TOKEN_DETAILS_READY",
            )
        )

        self._subscriptions.append(
            bus.create_subscription_to_pop_by_type(
                self._asset_details_event_type,
                self._on_asset_details_ready,
                name="cesium.omniverse.ASSET_DETAILS_READY",
            )
        )

    def _on_token_details_ready(self, _e: carb.events.IEvent):
        self._logger.info("Received token details event.")

        default_token_details = self._cesium_omniverse_interface.get_default_token_troubleshooting_details()

        if self._default_token_stack is not None:
            self._default_token_stack.visible = default_token_details.show_details

        if self._default_token_is_valid_widget is not None:
            self._default_token_is_valid_widget.passed = default_token_details.is_valid

        if self._default_token_has_access_widget is not None:
            self._default_token_has_access_widget.passed = default_token_details.allows_access_to_asset

        if self._default_token_associated_to_account_widget is not None:
            self._default_token_associated_to_account_widget.passed = (
                default_token_details.associated_with_user_account
            )

        asset_token_details = self._cesium_omniverse_interface.get_asset_token_troubleshooting_details()

        if self._asset_token_stack is not None:
            self._asset_token_stack.visible = asset_token_details.show_details

        if self._asset_token_is_valid_widget is not None:
            self._asset_token_is_valid_widget.passed = asset_token_details.is_valid

        if self._asset_token_has_access_widget is not None:
            self._asset_token_has_access_widget.passed = asset_token_details.allows_access_to_asset

        if self._asset_token_associated_to_account_widget is not None:
            self._asset_token_associated_to_account_widget.passed = asset_token_details.associated_with_user_account

    def _on_asset_details_ready(self, _e: carb.events.IEvent):
        asset_details = self._cesium_omniverse_interface.get_asset_troubleshooting_details()

        if self._asset_on_account_widget is not None:
            self._asset_on_account_widget.passed = asset_details.asset_exists_in_user_account

    @staticmethod
    def _on_open_ion_button_clicked():
        webbrowser.open("https://ion.cesium.com")

    def _build_ui(self):
        with ui.VStack(spacing=10):
            ui.Label(self._message, height=54, word_wrap=True)
            with ui.VGrid(spacing=10, column_count=2):
                self._asset_token_stack = ui.VStack(spacing=5, visible=False)
                with self._asset_token_stack:
                    ui.Label(
                        f"{self._name}'s Access Token",
                        height=16,
                        style=CesiumOmniverseUiStyles.troubleshooter_header_style,
                    )
                    with ui.HStack(height=16, spacing=10):
                        self._asset_token_is_valid_widget = CesiumPassFailWidget()
                        ui.Label("Is a valid Cesium ion Token")
                    with ui.HStack(height=16, spacing=10):
                        self._asset_token_has_access_widget = CesiumPassFailWidget()
                        ui.Label("Allows access to this asset")
                    with ui.HStack(height=16, spacing=10):
                        self._asset_token_associated_to_account_widget = CesiumPassFailWidget()
                        ui.Label("Is associated with your user account")
                self._default_token_stack = ui.VStack(spacing=5, visible=False)
                with self._default_token_stack:
                    ui.Label(
                        "Project Default Access Token",
                        height=16,
                        style=CesiumOmniverseUiStyles.troubleshooter_header_style,
                    )
                    with ui.HStack(height=16, spacing=10):
                        self._default_token_is_valid_widget = CesiumPassFailWidget()
                        ui.Label("Is a valid Cesium ion Token")
                    with ui.HStack(height=16, spacing=10):
                        self._default_token_has_access_widget = CesiumPassFailWidget()
                        ui.Label("Allows access to this asset")
                    with ui.HStack(height=16, spacing=10):
                        self._default_token_associated_to_account_widget = CesiumPassFailWidget()
                        ui.Label("Is associated with your user account")
                with ui.VStack(spacing=5):
                    ui.Label("Asset", height=16, style=CesiumOmniverseUiStyles.troubleshooter_header_style)
                    with ui.HStack(height=16, spacing=10):
                        self._asset_on_account_widget = CesiumPassFailWidget()
                        ui.Label("Asset ID exists in your user account")
            ui.Spacer()
            ui.Button(
                "Open Cesium ion on the Web",
                alignment=ui.Alignment.CENTER,
                height=36,
                style=CesiumOmniverseUiStyles.blue_button_style,
                clicked_fn=self._on_open_ion_button_clicked,
            )

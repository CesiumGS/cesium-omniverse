import logging
import carb.events
import omni.kit.app as app
import omni.ui as ui
import omni.kit.clipboard as clipboard
import webbrowser
from pathlib import Path
from typing import List, Optional
from ..bindings import ICesiumOmniverseInterface
from .styles import CesiumOmniverseUiStyles


class CesiumOmniverseSignInWidget(ui.Frame):
    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, **kwargs):
        manager = app.get_app().get_extension_manager()
        ext_id = manager.get_extension_id_by_module("cesium.omniverse")
        self._logger = logging.getLogger(__name__)
        self._images_path = Path(manager.get_extension_path(ext_id)).joinpath("images")
        self._cesium_omniverse_interface = cesium_omniverse_interface

        self._connect_button: Optional[ui.Button] = None
        self._waiting_message_frame: Optional[ui.Frame] = None
        self._authorize_url_field: Optional[ui.StringField] = None

        self._subscriptions: List[carb.events.ISubscription] = []
        self._setup_subscriptions()

        super().__init__(build_fn=self._build_ui, **kwargs)

    def destroy(self):
        for subscription in self._subscriptions:
            subscription.unsubscribe()
        self._subscriptions.clear()

    def _setup_subscriptions(self):
        update_stream = app.get_app().get_update_event_stream()
        self._subscriptions.append(
            update_stream.create_subscription_to_pop(self._on_update_frame, name="on_update_frame")
        )

    def _on_update_frame(self, _e: carb.events.IEvent):
        if not self.visible:
            return

        session = self._cesium_omniverse_interface.get_session()
        if session is not None and self._waiting_message_frame is not None:
            self._waiting_message_frame.visible = session.is_connecting()

            if session.is_connecting():
                authorize_url = session.get_authorize_url()

                if self._authorize_url_field.model.get_value_as_string() != authorize_url:
                    self._authorize_url_field.model.set_value(authorize_url)
                    webbrowser.open(authorize_url)

    def _build_ui(self):
        with self:
            with ui.VStack(alignment=ui.Alignment.CENTER_TOP, spacing=ui.Length(20, ui.UnitType.PIXEL)):
                ui.Spacer(height=0)
                ui.Image(
                    f"{self._images_path}/placeholder_logo.png",
                    alignment=ui.Alignment.CENTER,
                    fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT,
                    height=140,
                )
                with ui.HStack(height=0):
                    ui.Spacer()
                    ui.Label(
                        "Access global high-resolution 3D content, including photogrammetry, "
                        "terrain, imagery, and buildings. Bring your own data for tiling, hosting, "
                        "and streaming to Omniverse.",
                        alignment=ui.Alignment.CENTER,
                        style=CesiumOmniverseUiStyles.intro_label_style,
                        width=ui.Length(80, ui.UnitType.PERCENT),
                        word_wrap=True,
                    )
                    ui.Spacer()
                with ui.HStack(height=0):
                    ui.Spacer()
                    self._connect_button = ui.Button(
                        "Connect to Cesium ion",
                        alignment=ui.Alignment.CENTER,
                        height=ui.Length(36, ui.UnitType.PIXEL),
                        width=ui.Length(180, ui.UnitType.PIXEL),
                        style=CesiumOmniverseUiStyles.blue_button_style,
                        clicked_fn=self._connect_button_clicked,
                    )
                    ui.Spacer()
                self._waiting_message_frame = ui.Frame(visible=False, height=0)
                with self._waiting_message_frame:
                    with ui.VStack(spacing=10):
                        ui.Label("Waiting for you to sign into Cesium ion with your web browser...")
                        ui.Button("Open web browser again", clicked_fn=self._open_web_browser_again_clicked)
                        ui.Label("Or copy the URL below into your web browser.")
                        with ui.HStack():
                            self._authorize_url_field = ui.StringField(read_only=True)
                            self._authorize_url_field.model.set_value("https://cesium.com")
                            ui.Button("Copy to Clipboard", clicked_fn=self._copy_to_clipboard_clicked)
                ui.Spacer(height=10)

    def _connect_button_clicked(self) -> None:
        self._cesium_omniverse_interface.connect_to_ion()

    def _open_web_browser_again_clicked(self) -> None:
        webbrowser.open(self._authorize_url_field.model.get_value_as_string())

    def _copy_to_clipboard_clicked(self) -> None:
        if self._authorize_url_field is not None:
            clipboard.copy(self._authorize_url_field.model.get_value_as_string())

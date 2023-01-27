import logging
import omni.kit.app as app
import omni.ui as ui
import omni.kit.clipboard as clipboard
from pathlib import Path
from typing import Optional
from .styles import CesiumOmniverseUiStyles


class CesiumOmniverseSignInWidget(ui.Frame):
    def __init__(self, **kwargs):
        manager = app.get_app().get_extension_manager()
        ext_id = manager.get_extension_id_by_module("cesium.omniverse")
        self._logger = logging.getLogger(__name__)
        self._images_path = Path(manager.get_extension_path(ext_id)).joinpath("images")

        self._connect_button: Optional[ui.Button] = None
        self._waiting_message_frame: Optional[ui.Frame] = None
        self._url_string_model: Optional[ui.StringField] = None

        super().__init__(build_fn=self._build_ui, **kwargs)

    def _build_ui(self):
        with self:
            with ui.VStack(alignment=ui.Alignment.CENTER_TOP, spacing=ui.Length(20, ui.UnitType.PIXEL)):
                ui.Image(f"{self._images_path}/placeholder_logo.png", alignment=ui.Alignment.CENTER,
                         fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT, height=ui.Length(25, ui.UnitType.PERCENT))
                with ui.HStack(height=0):
                    ui.Spacer()
                    ui.Label(
                        "Access global high-resolution 3D content, including photogrammetry, "
                        "terrain, imagery, and buildings. Bring your own data for tiling, hosting, "
                        "and streaming to Omniverse.",
                        alignment=ui.Alignment.CENTER,
                        style=CesiumOmniverseUiStyles.intro_label_style,
                        width=ui.Length(80, ui.UnitType.PERCENT),
                        word_wrap=True
                    )
                    ui.Spacer()
                with ui.HStack(height=0):
                    ui.Spacer()
                    self._connect_button = ui.Button("Connect to Cesium ion",
                                                     alignment=ui.Alignment.CENTER,
                                                     height=ui.Length(36, ui.UnitType.PIXEL),
                                                     width=ui.Length(180, ui.UnitType.PIXEL),
                                                     style=CesiumOmniverseUiStyles.blue_button_style,
                                                     clicked_fn=self._connect_button_clicked)
                    ui.Spacer()
                self._waiting_message_frame = ui.Frame(visible=False, height=0)
                with self._waiting_message_frame:
                    with ui.VStack(spacing=10):
                        ui.Label("Waiting for you to sign into Cesium ion with your web browser...")
                        ui.Button("Open web browser again", clicked_fn=self._open_web_browser_again_clicked)
                        ui.Label("Or copy the URL below into your web browser.")
                        with ui.HStack():
                            self._url_string_field = ui.StringField(read_only=True)
                            self._url_string_field.model.set_value("https://www.cesium.com")
                            ui.Button("Copy to Clipboard", clicked_fn=self._copy_to_clipboard_clicked)
                ui.Spacer(height=10)

    def _connect_button_clicked(self) -> None:
        # TODO: Actually handle updating the message frame to show once we have the URL.
        self._handle_show_waiting_message_frame()

        # TODO: Call cpp code to start sign-in process.

        pass

    def _handle_show_waiting_message_frame(self) -> None:
        self._waiting_message_frame.visible = True

    def _open_web_browser_again_clicked(self) -> None:
        pass

    def _copy_to_clipboard_clicked(self) -> None:
        if self._url_string_field is not None:
            clipboard.copy("foobar")

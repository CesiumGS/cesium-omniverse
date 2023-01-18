from ..bindings import ICesiumOmniverseInterface
import logging
import omni.ui as ui
from typing import Optional


class CesiumOmniverseMainWindow(ui.Window):
    """
    The main window for working with Cesium for Omniverse. Docked in the same area as "Stage".
    """

    WINDOW_NAME = "Cesium"
    MENU_PATH = f"Window/Cesium/{WINDOW_NAME}"

    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, **kwargs):
        super().__init__(CesiumOmniverseMainWindow.WINDOW_NAME, **kwargs)

        self._cesium_omniverse_interface = cesium_omniverse_interface
        self._logger = logging.getLogger(__name__)

        # Buttons aren't created until the build function is called.
        self._add_button: Optional[ui.Button] = None
        self._upload_button: Optional[ui.Button] = None
        self._token_button: Optional[ui.Button] = None
        self._learn_button: Optional[ui.Button] = None
        self._help_button: Optional[ui.Button] = None
        self._sign_out_button: Optional[ui.Button] = None

        self.frame.set_build_fn(self._build_fn)

    def destroy(self) -> None:
        super().destroy()

    def _build_fn(self):
        """Builds all UI components."""

        with ui.VStack():
            with ui.HStack():
                self._add_button = ui.Button("Add", clicked_fn=self._add_button_clicked, enabled=False)
                self._upload_button = ui.Button("Upload", clicked_fn=self._upload_button_clicked, enabled=False)
                self._token_button = ui.Button("Token", clicked_fn=self._token_button_clicked)
                self._learn_button = ui.Button("Learn", clicked_fn=self._learn_button_clicked)
                self._help_button = ui.Button("Help", clicked_fn=self._help_button_clicked)
                self._sign_out_button = ui.Button("Sign Out", clicked_fn=self._sign_out_button_clicked, enabled=False)
            ui.Label("TODO: The rest of this window.")

    def _add_button_clicked(self) -> None:
        if not self._add_button or not self._add_button.enabled:
            return

        # TODO: Implement CesiumMainWindow._add_button_clicked(self)

        pass

    def _upload_button_clicked(self) -> None:
        if not self._upload_button or not self._upload_button.enabled:
            return

        # TODO: Implement CesiumMainWindow._upload_button_clicked(self)

        pass

    def _token_button_clicked(self) -> None:
        if not self._token_button:
            return

        # TODO: Implement CesiumMainWindow._token_button_clicked(self)

        pass

    def _learn_button_clicked(self) -> None:
        if not self._learn_button:
            return

        # TODO: Implement CesiumMainWindow._learn_button_clicked(self)

        pass

    def _help_button_clicked(self) -> None:
        if not self._help_button:
            return

        # TODO: Implement CesiumMainWindow._help_button_clicked(self)

        pass

    def _sign_out_button_clicked(self) -> None:
        if not self._sign_out_button or not self._upload_button.enabled:
            return

        # TODO: Implement CesiumMainWindow._sign_out_button_clicked(self)

        pass

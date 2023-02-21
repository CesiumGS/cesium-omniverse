from typing import Optional

import omni.ui as ui
from .models import IonAssetItem
from .styles import CesiumOmniverseUiStyles


class CesiumAssetDetailsWidget(ui.ScrollingFrame):
    def __init__(self, asset: Optional[IonAssetItem] = None, **kwargs):
        super().__init__(**kwargs)

        self.style = CesiumOmniverseUiStyles.asset_detail_frame

        self._name = asset.name.as_string if asset else ""
        self._id = asset.id.as_int if asset else 0
        self._description = asset.description.as_string if asset else ""
        self._attribution = asset.attribution.as_string if asset else ""

        self._name_label: Optional[ui.Label] = None
        self._id_label: Optional[ui.Label] = None
        self._description_label: Optional[ui.Label] = None
        self._attribution_label: Optional[ui.Label] = None

        self.set_build_fn(self._build_fn)

    def __del__(self):
        self.destroy()

    def destroy(self) -> None:
        if self._name_label is not None:
            self._name_label.destroy()

        if self._id_label is not None:
            self._id_label.destroy()

        if self._description_label is not None:
            self._description_label.destroy()

        if self._attribution_label is not None:
            self._attribution_label.destroy()

    def update_selection(self, asset: Optional[IonAssetItem]):
        self._name = asset.name.as_string if asset else ""
        self._id = asset.id.as_int if asset else 0
        self._description = asset.description.as_string if asset else ""
        self._attribution = asset.attribution.as_string if asset else ""

        self.rebuild()

    def _should_be_visible(self):
        return self._name != "" or self._id != 0 or self._description != "" or self._attribution != ""

    def _build_fn(self):
        with self:
            if self._should_be_visible():
                with ui.VStack(spacing=20):
                    with ui.VStack(spacing=5):
                        ui.Label(self._name, style=CesiumOmniverseUiStyles.asset_detail_name_label, height=0)
                        ui.Label(f"(ID: {self._id})", style=CesiumOmniverseUiStyles.asset_detail_id_label, height=0)
                    # TODO: Add to stage buttons.
                    with ui.VStack(spacing=5):
                        ui.Label("Description", style=CesiumOmniverseUiStyles.asset_detail_header_label, height=0)
                        ui.Label(self._description, word_wrap=True, alignment=ui.Alignment.TOP, height=0)
                    with ui.VStack(spacing=5):
                        ui.Label("Attribution", style=CesiumOmniverseUiStyles.asset_detail_header_label, height=0)
                        ui.Label(self._attribution, word_wrap=True, alignment=ui.Alignment.TOP, height=0)
            else:
                ui.Spacer()

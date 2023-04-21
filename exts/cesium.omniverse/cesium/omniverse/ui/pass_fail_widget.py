from pathlib import Path
import carb.events
import omni.kit.app as app
import omni.ui as ui
from typing import List


class CesiumPassFailWidget(ui.Frame):
    def __init__(self, passed=False, **kwargs):
        self._passed_model = ui.SimpleBoolModel(passed)

        manager = app.get_app().get_extension_manager()
        ext_id = manager.get_extension_id_by_module("cesium.omniverse")
        self._icon_path = Path(manager.get_extension_path(ext_id)).joinpath("images")

        self._subscriptions: List[carb.Subscription] = []
        self._setup_subscriptions()

        super().__init__(build_fn=self._build_ui, **kwargs)

    def __del__(self):
        self.destroy()

    def destroy(self):
        for subscription in self._subscriptions:
            subscription.unsubscribe()
        self._subscriptions.clear()

    @property
    def passed(self) -> bool:
        return self._passed_model.get_value_as_bool()

    @passed.setter
    def passed(self, value: bool):
        self._passed_model.set_value(value)

    def _setup_subscriptions(self):
        self._subscriptions.append(self._passed_model.subscribe_value_changed_fn(lambda _e: self.rebuild()))

    def _build_ui(self):
        with self:
            with ui.VStack(width=16, height=16):
                path_root = f"{self._icon_path}/FontAwesome"
                icon = (
                    f"{path_root}/check-solid.svg"
                    if self._passed_model.get_value_as_bool()
                    else f"{path_root}/times-solid.svg"
                )

                ui.Image(
                    icon,
                    fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT,
                    alignment=ui.Alignment.CENTER_BOTTOM,
                    width=16,
                    height=16,
                )

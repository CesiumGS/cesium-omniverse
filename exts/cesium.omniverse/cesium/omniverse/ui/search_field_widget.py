from typing import Callable, List
import carb.events
import omni.ui as ui


class CesiumSearchFieldWidget(ui.Frame):
    def __init__(
        self, callback_fn: Callable[[ui.AbstractValueModel], None], default_value="", font_size=16, **kwargs
    ):
        self._callback_fn = callback_fn
        self._search_value = ui.SimpleStringModel(default_value)
        self._font_size = font_size

        self._subscriptions: List[carb.Subscription] = []
        self._setup_subscriptions()

        super().__init__(build_fn=self._build_fn, **kwargs)

    def destroy(self):
        super().destroy()

    @property
    def search_value(self) -> str:
        return self._search_value.get_value_as_string()

    @search_value.setter
    def search_value(self, value: str):
        self._search_value.set_value(value)

    def _setup_subscriptions(self):
        self._subscriptions.append(self._search_value.subscribe_value_changed_fn(self._callback_fn))

    def _build_fn(self):
        with self:
            with ui.ZStack():
                ui.StringField(model=self._search_value, style={"font_size": self._font_size})
                with ui.HStack(alignment=ui.Alignment.CENTER):
                    ui.Spacer()
                    image_size = self._font_size * 2
                    ui.Image(
                        "resources/glyphs/menu_search.svg", width=image_size, height=image_size, style={"margin": 4}
                    )

from typing import Callable, List, Optional
import carb.events
import omni.ui as ui
from omni.ui import color as cl


class CesiumSearchFieldWidget(ui.Frame):
    def __init__(
        self, callback_fn: Callable[[ui.AbstractValueModel], None], default_value="", font_size=14, **kwargs
    ):
        self._callback_fn = callback_fn
        self._search_value = ui.SimpleStringModel(default_value)
        self._font_size = font_size
        self._clear_button_stack: Optional[ui.Stack] = None

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
        self._set_clear_button_visibility()

    def _update_visibility(self, _e):
        self._set_clear_button_visibility()

    def _setup_subscriptions(self):
        self._subscriptions.append(self._search_value.subscribe_value_changed_fn(self._callback_fn))
        self._subscriptions.append(self._search_value.subscribe_value_changed_fn(self._update_visibility))

    def _on_clear_click(self):
        self._search_value.set_value("")
        self._set_clear_button_visibility()

    def _set_clear_button_visibility(self):
        self._clear_button_stack.visible = self._search_value.as_string != ""

    def _build_fn(self):
        with self:
            with ui.ZStack(height=0):
                ui.Rectangle(style={"background_color": cl("#1F2123"), "border_radius": 3})
                with ui.HStack(alignment=ui.Alignment.CENTER):
                    image_size = self._font_size * 2
                    ui.Image(
                        "resources/glyphs/menu_search.svg",
                        width=image_size,
                        height=image_size,
                        style={"margin": 4},
                    )
                    with ui.VStack():
                        ui.Spacer()
                        ui.StringField(
                            model=self._search_value, height=self._font_size, style={"font_size": self._font_size}
                        )
                        ui.Spacer()
                    self._clear_button_stack = ui.VStack(width=0, visible=False)
                    with self._clear_button_stack:
                        ui.Spacer()
                        ui.Button(
                            image_url="resources/icons/Close.png",
                            width=0,
                            height=0,
                            image_width=self._font_size,
                            image_height=self._font_size,
                            style={"margin": 4, "background_color": cl("#1F2123")},
                            clicked_fn=self._on_clear_click,
                            opaque_for_mouse_events=True,
                        )
                        ui.Spacer()

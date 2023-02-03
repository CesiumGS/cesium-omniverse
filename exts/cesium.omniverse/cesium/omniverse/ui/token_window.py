import logging
import carb.events
import omni.kit.app as app
import omni.ui as ui
from enum import Enum
from typing import List, Optional
from ..bindings import ICesiumOmniverseInterface, Token
from .styles import CesiumOmniverseUiStyles

SELECT_TOKEN_TEXT = "Cesium for Omniverse embeds a Cesium ion token in your stage in order to allow it " \
                    "to access the assets you add. Select the Cesium ion token to use."
CREATE_NEW_LABEL_TEXT = "Create a new token"
USE_EXISTING_LABEL_TEXT = "Use an existing token"
SPECIFY_TOKEN_LABEL_TEXT = "Specify a token"
CREATE_NEW_FIELD_LABEL_TEXT = "Name"
USE_EXISTING_FIELD_LABEL_TEXT = "Token"
SPECIFY_TOKEN_FIELD_LABEL_TEXT = "Token"
SELECT_BUTTON_TEXT = "Use as Project Default Token"

OUTER_SPACING = 10
CHECKBOX_WIDTH = 20
INNER_HEIGHT = 20
FIELD_SPACING = 8
FIELD_LABEL_WIDTH = 40


class TokenOptionEnum(Enum):
    CREATE_NEW = 1
    USE_EXISTING = 2
    SPECIFY_TOKEN = 3


class UseExistingComboItem(ui.AbstractItem):
    def __init__(self, token: Token):
        super().__init__()
        self.id = ui.SimpleStringModel(token.id)
        self.name = ui.SimpleStringModel(token.name)
        self.token = ui.SimpleStringModel(token.token)


class UseExistingComboModel(ui.AbstractItemModel):
    def __init__(self, item_list):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._current_index = ui.SimpleIntModel(0)
        self._current_index.add_value_changed_fn(lambda index_model: self._item_changed(None))

        self._items = [UseExistingComboItem(text) for text in item_list]

    def replace_all_items(self, items: List[str]):
        self._items.clear()
        self._items = [UseExistingComboItem(text) for text in items]
        self._current_index.set_value(0)
        self._item_changed(None)

    def append_child_item(self, parent_item: ui.AbstractItem, model: str):
        self._items.append(UseExistingComboItem(model))
        self._item_changed(None)

    def get_item_children(self, item=None):
        return self._items

    def get_item_value_model(self, item: UseExistingComboItem = None, column_id: int = 0):
        if item is None:
            return self._current_index
        return item.name


class CesiumOmniverseTokenWindow(ui.Window):
    WINDOW_NAME = "Select Cesium ion Token"

    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, **kwargs):
        super().__init__(CesiumOmniverseTokenWindow.WINDOW_NAME, **kwargs)

        self._cesium_omniverse_interface = cesium_omniverse_interface
        self._logger = logging.getLogger(__name__)

        self.height = 400
        self.width = 600

        self.padding_x = 12
        self.padding_y = 12

        self._selected_option = TokenOptionEnum.CREATE_NEW
        self._create_new_radio_button_model = ui.SimpleBoolModel(True)
        self._create_new_radio_button_model.add_value_changed_fn(
            lambda m: self._radio_button_changed(m, TokenOptionEnum.CREATE_NEW))
        self._use_existing_radio_button_model = ui.SimpleBoolModel(False)
        self._use_existing_radio_button_model.add_value_changed_fn(
            lambda m: self._radio_button_changed(m, TokenOptionEnum.USE_EXISTING))
        self._specify_token_radio_button_model = ui.SimpleBoolModel(False)
        self._specify_token_radio_button_model.add_value_changed_fn(
            lambda m: self._radio_button_changed(m, TokenOptionEnum.SPECIFY_TOKEN))
        self._use_existing_combo_box: Optional[ui.ComboBox] = None
        self._create_new_field_model = ui.SimpleStringModel()
        self._use_existing_combo_model = UseExistingComboModel([])
        self._specify_token_field_model = ui.SimpleStringModel()
        self._reload_next_frame = False

        self._subscriptions: List[carb.events.ISubscription] = []
        self._setup_subscriptions()

        self.frame.set_build_fn(self._build_fn)

    def destroy(self):
        for subscription in self._subscriptions:
            subscription.unsubscribe()

        super().destroy()

    def _setup_subscriptions(self):
        update_stream = app.get_app().get_update_event_stream()

        self._subscriptions.append(
            update_stream.create_subscription_to_pop(self._on_update_frame, name="on_update_frame")
        )

    def _on_update_frame(self, _e: carb.events.IEvent):
        session = self._cesium_omniverse_interface.get_session()

        if self._reload_next_frame and session.is_token_list_loaded():
            token_list = session.get_tokens()
            self._use_existing_combo_model.replace_all_items(token_list)
            if self._use_existing_combo_box:
                self._use_existing_combo_box.enabled = True
            self._reload_next_frame = False
        elif session.is_loading_token_list():
            if self._use_existing_combo_box:
                self._use_existing_combo_box.enabled = False
            self._reload_next_frame = True

    def _select_button_clicked(self):
        pass

    def _radio_button_changed(self, model, selected: TokenOptionEnum):
        if not model.get_value_as_bool():
            self._reselect_if_none_selected(selected)
            return

        self._selected_option = selected

        if self._selected_option is TokenOptionEnum.CREATE_NEW:
            self._create_new_radio_button_model.set_value(True)
            self._use_existing_radio_button_model.set_value(False)
            self._specify_token_radio_button_model.set_value(False)
        elif self._selected_option is TokenOptionEnum.USE_EXISTING:
            self._create_new_radio_button_model.set_value(False)
            self._use_existing_radio_button_model.set_value(True)
            self._specify_token_radio_button_model.set_value(False)
        elif self._selected_option is TokenOptionEnum.SPECIFY_TOKEN:
            self._create_new_radio_button_model.set_value(False)
            self._use_existing_radio_button_model.set_value(False)
            self._specify_token_radio_button_model.set_value(True)

    def _reselect_if_none_selected(self, selected):
        """
        Reselect the checkbox in our "radio buttons" if none are selected. This is necessary because Omniverse
        has us using checkboxes to make a radio button rather than a proper radio button.
        """

        if not self._create_new_radio_button_model.get_value_as_bool() and \
                not self._use_existing_radio_button_model.get_value_as_bool() \
                and not self._specify_token_radio_button_model.get_value_as_bool():
            if selected is TokenOptionEnum.CREATE_NEW:
                self._create_new_radio_button_model.set_value(True)
            elif selected is TokenOptionEnum.USE_EXISTING:
                self._use_existing_radio_button_model.set_value(True)
            elif selected is TokenOptionEnum.SPECIFY_TOKEN:
                self._specify_token_radio_button_model.set_value(True)

    @staticmethod
    def _build_field(label_text: str, field_label_text: str, checkbox_model: ui.SimpleBoolModel,
                     string_field_model: ui.SimpleStringModel):
        with ui.HStack(spacing=OUTER_SPACING):
            ui.CheckBox(checkbox_model, width=CHECKBOX_WIDTH)
            with ui.VStack(height=INNER_HEIGHT, spacing=FIELD_SPACING):
                ui.Label(label_text)
                with ui.HStack():
                    ui.Label(field_label_text, width=FIELD_LABEL_WIDTH)
                    ui.StringField(string_field_model)

    def _build_fn(self):
        with ui.VStack(spacing=10):
            ui.Label(SELECT_TOKEN_TEXT, word_wrap=True)
            self._build_field(CREATE_NEW_LABEL_TEXT, CREATE_NEW_FIELD_LABEL_TEXT, self._create_new_radio_button_model,
                              self._create_new_field_model)
            with ui.HStack(spacing=OUTER_SPACING):
                ui.CheckBox(self._use_existing_radio_button_model, width=CHECKBOX_WIDTH)
                with ui.VStack(height=20, spacing=FIELD_SPACING):
                    ui.Label(USE_EXISTING_LABEL_TEXT)
                    with ui.HStack():
                        ui.Label(USE_EXISTING_FIELD_LABEL_TEXT, width=FIELD_LABEL_WIDTH)
                        self._use_existing_combo_box = ui.ComboBox(self._use_existing_combo_model)
            self._build_field(SPECIFY_TOKEN_LABEL_TEXT, SPECIFY_TOKEN_FIELD_LABEL_TEXT,
                              self._specify_token_radio_button_model, self._specify_token_field_model)
            ui.Button(SELECT_BUTTON_TEXT, alignment=ui.Alignment.CENTER, height=36,
                      style=CesiumOmniverseUiStyles.blue_button_style, clicked_fn=self._select_button_clicked)

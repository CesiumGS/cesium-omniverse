import logging
import carb.events
import omni.kit.app as app
import omni.ui as ui
from typing import List, Optional
from datetime import datetime
from ..bindings import ICesiumOmniverseInterface
from .styles import CesiumOmniverseUiStyles


class DateModel(ui.AbstractValueModel):
    """Takes an RFC 3339 formatted timestamp and produces a date value."""

    def __init__(self, value: str):
        super().__init__()
        self._value = datetime.strptime(value[0:19], "%Y-%m-%dT%H:%M:%S")

    def get_value_as_string(self) -> str:
        if self._value is None:
            return ""

        return self._value.strftime("%Y-%m-%d")


class IonAssetItem(ui.AbstractItem):
    """Represents an ion Asset."""

    def __init__(self, asset_id: int, name: str, description: str, attribution: str, asset_type: str, date_added: str):
        super().__init__()
        self.id = ui.SimpleIntModel(asset_id)
        self.name = ui.SimpleStringModel(name)
        self.description = ui.SimpleStringModel(description)
        self.attribution = ui.SimpleStringModel(attribution)
        self.type = ui.SimpleStringModel(asset_type)
        self.dateAdded = DateModel(date_added)

    def __repr__(self):
        return f"{self.name.as_string} (ID: {self.id.as_int})"


class IonAssets(ui.AbstractItemModel):
    """Represents a list of ion assets for the asset window."""

    def __init__(self, items=None):
        super().__init__()
        if items is None:
            items = []
        self._items: List[IonAssetItem] = items

    def replace_items(self, items: List[IonAssetItem]):
        self._items.clear()
        self._items.extend(items)
        self._item_changed(None)

    def get_item_children(self, item: IonAssetItem = None) -> List[IonAssetItem]:
        if item is not None:
            return []

        return self._items

    def get_item_value_model_count(self, item: IonAssetItem = None) -> int:
        """The number of columns"""
        return 3

    def get_item_value_model(self, item: IonAssetItem = None, column_id: int = 0) -> ui.AbstractValueModel:
        """Returns the value model for the specific column."""

        if item is None:
            item = self._items[0]

        # When we are finally on Python 3.10 with Omniverse, we should change this to a switch.
        return item.name if column_id == 0 else item.type if column_id == 1 else item.dateAdded


class IonAssetDelegate(ui.AbstractItemDelegate):

    def build_header(self, column_id: int = 0) -> None:
        with ui.ZStack(height=20):
            if column_id == 0:
                ui.Label("Name")
            elif column_id == 1:
                ui.Label("Type")
            else:
                ui.Label("Date Added")

    def build_branch(self, model: ui.AbstractItemModel, item: ui.AbstractItem = None, column_id: int = 0,
                     level: int = 0,
                     expanded: bool = False) -> None:
        # We don't use this because we don't have a hierarchy, but we need to at least stub it out.
        pass

    def build_widget(self, model: IonAssets, item: IonAssetItem = None, column_id: int = 0, level: int = 0,
                     expanded: bool = False) -> None:
        with ui.ZStack(height=20):
            value_model = model.get_item_value_model(item, column_id)
            ui.Label(value_model.as_string)


class CesiumOmniverseAssetWindow(ui.Window):
    """
    The asset list window for Cesium for Omniverse. Docked in the same area as "Assets".
    """

    WINDOW_NAME = "Cesium Assets"
    MENU_PATH = f"Window/Cesium/{WINDOW_NAME}"

    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, **kwargs):
        super().__init__(CesiumOmniverseAssetWindow.WINDOW_NAME, **kwargs)

        self._cesium_omniverse_interface = cesium_omniverse_interface
        self._logger = logging.getLogger(__name__)

        self._assets = IonAssets()
        self._assets_delegate = IonAssetDelegate()

        self._refresh_button: Optional[ui.Button] = None

        self._subscriptions: List[carb.events.ISubscription] = []
        self._setup_subscriptions()

        self.frame.set_build_fn(self._build_fn)

        self._refresh_list()

        self.focus()

    def destroy(self):
        for subscription in self._subscriptions:
            subscription.unsubscribe()
        self._subscriptions.clear()

        super().destroy()

    def _setup_subscriptions(self):
        bus = app.get_app().get_message_bus_event_stream()

        assets_updated_event = carb.events.type_from_string("cesium.omniverse.ASSETS_UPDATED")
        self._subscriptions.append(
            bus.create_subscription_to_pop_by_type(assets_updated_event, self._on_assets_updated,
                                                   name="cesium.omniverse.asset_window.assets_updated")
        )

    def _refresh_list(self):
        session = self._cesium_omniverse_interface.get_session()

        if session is not None:
            self._logger.info("Cesium ion Assets refreshing.")
            session.refresh_assets()

    def _on_assets_updated(self, _e: carb.events.IEvent):
        session = self._cesium_omniverse_interface.get_session()

        if session is not None:
            self._logger.info("Cesium ion Assets refreshed.")
            self._assets.replace_items(
                [
                    IonAssetItem(
                        item.asset_id,
                        item.name,
                        item.description,
                        item.attribution,
                        item.asset_type,
                        item.date_added) for item in session.get_assets().items
                ]
            )

    def _refresh_button_clicked(self):
        self._refresh_list()

    def _build_fn(self):
        """Builds all UI components."""

        with ui.VStack(spacing=5):
            with ui.HStack(height=30):
                self._refresh_button = ui.Button("Refresh", alignment=ui.Alignment.CENTER, width=80,
                                                 style=CesiumOmniverseUiStyles.blue_button_style,
                                                 clicked_fn=self._refresh_button_clicked)
                ui.Spacer()
            with ui.HStack(spacing=5):
                with ui.ScrollingFrame(style_type_name_override="TreeView",
                                       style={"Field": {"background_color": 0xFF000000}},
                                       width=ui.Length(2, ui.UnitType.FRACTION)):
                    ui.TreeView(self._assets, delegate=self._assets_delegate, root_visible=False, header_visible=True,
                                style={"TreeView.Item": {"margin": 4}})
                with ui.ScrollingFrame(width=ui.Length(1, ui.UnitType.FRACTION)):
                    with ui.VStack():
                        ui.Label("TODO: Selection Frame")

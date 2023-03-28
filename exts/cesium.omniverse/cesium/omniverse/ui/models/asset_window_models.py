from typing import List

import omni.ui as ui
from .date_model import DateModel


class IonAssetItem(ui.AbstractItem):
    """Represents an ion Asset."""

    def __init__(
        self, asset_id: int, name: str, description: str, attribution: str, asset_type: str, date_added: str
    ):
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

    def __init__(self, items=None, filter_value=""):
        super().__init__()
        if items is None:
            items = []
        self._items: List[IonAssetItem] = items
        self._visible_items: List[IonAssetItem] = []
        self._current_filter = filter_value
        self.filter_items(filter_value)

    def replace_items(self, items: List[IonAssetItem]):
        self._items.clear()
        self._items.extend(items)
        self.filter_items(self._current_filter)

    def filter_items(self, filter_value: str):
        self._current_filter = filter_value

        if filter_value == "":
            self._visible_items = self._items.copy()
        else:
            self._visible_items = [
                item for item in self._items if filter_value.lower() in item.name.as_string.lower()
            ]

        self._item_changed(None)

    def get_item_children(self, item: IonAssetItem = None) -> List[IonAssetItem]:
        if item is not None:
            return []

        return self._visible_items

    def get_item_value_model_count(self, item: IonAssetItem = None) -> int:
        """The number of columns"""
        return 3

    def get_item_value_model(self, item: IonAssetItem = None, column_id: int = 0) -> ui.AbstractValueModel:
        """Returns the value model for the specific column."""

        if item is None:
            item = self._visible_items[0]

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

    def build_branch(
        self,
        model: ui.AbstractItemModel,
        item: ui.AbstractItem = None,
        column_id: int = 0,
        level: int = 0,
        expanded: bool = False,
    ) -> None:
        # We don't use this because we don't have a hierarchy, but we need to at least stub it out.
        pass

    def build_widget(
        self, model: IonAssets, item: IonAssetItem = None, column_id: int = 0, level: int = 0, expanded: bool = False
    ) -> None:
        with ui.ZStack(height=20):
            value_model = model.get_item_value_model(item, column_id)
            ui.Label(value_model.as_string)

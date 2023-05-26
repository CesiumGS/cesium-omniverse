from typing import List
import omni.ui as ui


class StatisticsItem(ui.AbstractItem):
    """Represents a statistics item."""

    def __init__(self, label: str, value: ui.AbstractValueModel):
        super().__init__()
        self.label = ui.SimpleStringModel(label)
        self.value = value

    def __repr__(self):
        return f"{self.label}: {self.value.as_string}"


class StatisticsList(ui.AbstractItemModel):
    """Represents a list of statistics values for display."""

    def __init__(self, items=None):
        super().__init__()
        if items is None:
            items = []
        self._items: List[StatisticsItem] = items
        self._visible_items: List[StatisticsItem] = []

    def refresh(self):
        for item in self._items:
            self._item_changed(item)

    def get_item_children(self, item: StatisticsItem = None) -> List[StatisticsItem]:
        if item is not None:
            return []

        return self._items

    def get_item_value_model_count(self, item: StatisticsItem = None) -> int:
        """The number of columns."""
        return 2

    def get_item_value_model(self, item: StatisticsItem = None, column_id: int = 0) -> ui.AbstractValueModel:
        if item is None:
            item = self._items[0]

        # TODO: Move to a switch once we're on python 3.10
        return item.label if column_id == 0 else item.value


class StatisticsDelegate(ui.AbstractItemDelegate):
    def build_header(self, column_id: int = 0) -> None:
        if column_id == 0:
            ui.Label("Statistics")
        else:
            pass

    def build_branch(
        self,
        model: ui.AbstractItemModel,
        item: ui.AbstractItem = None,
        column_id: int = 0,
        level: int = 0,
        expanded: bool = False,
    ) -> None:
        pass

    def build_widget(
        self,
        model: ui.AbstractItemModel,
        item: StatisticsItem = None,
        column_id: int = 0,
        level: int = 0,
        expanded: bool = False,
    ) -> None:
        with ui.ZStack(height=20):
            value_model = model.get_item_value_model(item, column_id)
            ui.Label(value_model.get_value_as_string())

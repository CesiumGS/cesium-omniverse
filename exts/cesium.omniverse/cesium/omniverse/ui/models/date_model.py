from datetime import datetime
import omni.ui as ui


class DateModel(ui.AbstractValueModel):
    """Takes an RFC 3339 formatted timestamp and produces a date value."""

    def __init__(self, value: str):
        super().__init__()
        self._value = datetime.strptime(value[0:19], "%Y-%m-%dT%H:%M:%S")

    def get_value_as_string(self) -> str:
        return self._value.strftime("%Y-%m-%d")

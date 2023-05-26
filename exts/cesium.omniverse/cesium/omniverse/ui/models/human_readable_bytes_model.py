import logging
import omni.ui as ui


class HumanReadableBytesModel(ui.AbstractValueModel):
    """Takes an integer containing bytes and outputs a human-readable string."""

    def __init__(self, value: int):
        super().__init__()
        self._value = value

    def __str__(self):
        return self.get_value_as_string()

    def set_value(self, value: int):
        self._value = value

    def get_value_as_bool(self) -> bool:
        raise NotImplementedError

    def get_value_as_int(self) -> int:
        return self._value

    def get_value_as_float(self) -> float:
        return float(self._value)

    def get_value_as_string(self) -> str:
        for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB"]:
            logging.getLogger(__name__).warn(self._value)
            if abs(self._value) < 1024:
                return f"{self._value:3.2f} {unit}"
            self._value /= 1024
        return f"{self._value:.2f}YiB"

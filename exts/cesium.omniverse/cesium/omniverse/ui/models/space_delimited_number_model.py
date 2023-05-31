import math
import omni.ui as ui


class SpaceDelimitedNumberModel(ui.AbstractValueModel):
    """Divides large numbers into delimited groups for readability."""

    def __init__(self, value: float):
        super().__init__()
        self._value = value

    def __str__(self):
        return self.get_value_as_string()

    def set_value(self, value: float) -> None:
        self._value = value
        self._value_changed()

    def get_value_as_bool(self) -> bool:
        raise NotImplementedError

    def get_value_as_int(self) -> int:
        return math.trunc(self._value)

    def get_value_as_float(self) -> float:
        return self._value

    def get_value_as_string(self) -> str:
        # Replacing the comma with spaces because NIST did a lot of research and recommends it.
        #   https://physics.nist.gov/cuu/pdf/sp811.pdf#10.5.3
        return f"{self._value:,}".replace(",", " ")

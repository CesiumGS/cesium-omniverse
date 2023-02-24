from __future__ import annotations
from typing import Optional
import carb.events


class ImageryToAdd:
    def __init__(self, tileset_ion_id: int, imagery_ion_id: int, imagery_name: str):
        self.tileset_ion_id = tileset_ion_id
        self.imagery_ion_id = imagery_ion_id
        self.imagery_name = imagery_name

    def to_dict(self) -> dict:
        return {
            "tileset_ion_id": self.tileset_ion_id,
            "imagery_ion_id": self.imagery_ion_id,
            "imagery_name": self.imagery_name
        }

    @staticmethod
    def from_event(event: carb.events.IEvent) -> Optional[ImageryToAdd]:
        if event.payload is None or len(event.payload) == 0:
            return None

        return ImageryToAdd(event.payload["tileset_ion_id"],
                            event.payload["imagery_ion_id"],
                            event.payload["imagery_name"])

from __future__ import annotations
from typing import Optional
import carb.events


class ImageryToAdd:
    def __init__(self, tileset_path: str, imagery_ion_asset_id: int, imagery_name: str):
        self.tileset_path = tileset_path
        self.imagery_ion_asset_id = imagery_ion_asset_id
        self.imagery_name = imagery_name

    def to_dict(self) -> dict:
        return {
            "tileset_path": self.tileset_path,
            "imagery_ion_asset_id": self.imagery_ion_asset_id,
            "imagery_name": self.imagery_name,
        }

    @staticmethod
    def from_event(event: carb.events.IEvent) -> Optional[ImageryToAdd]:
        if event.payload is None or len(event.payload) == 0:
            return None

        return ImageryToAdd(
            event.payload["tileset_path"], event.payload["imagery_ion_asset_id"], event.payload["imagery_name"]
        )

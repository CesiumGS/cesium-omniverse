from __future__ import annotations
from typing import Optional
import carb.events


class AssetToAdd:
    def __init__(
        self,
        tileset_name: str,
        tileset_ion_id: int,
        imagery_name: Optional[str] = None,
        imagery_ion_id: Optional[int] = None,
    ):
        self.tileset_name = tileset_name
        self.tileset_ion_id = tileset_ion_id
        self.imagery_name = imagery_name
        self.imagery_ion_id = imagery_ion_id

    def to_dict(self) -> dict:
        return {
            "tileset_name": self.tileset_name,
            "tileset_ion_id": self.tileset_ion_id,
            "imagery_name": self.imagery_name,
            "imagery_ion_id": self.imagery_ion_id,
        }

    @staticmethod
    def from_event(event: carb.events.IEvent) -> Optional[AssetToAdd]:
        if event.payload is None or len(event.payload) == 0:
            return None

        return AssetToAdd(
            event.payload["tileset_name"],
            event.payload["tileset_ion_id"],
            event.payload["imagery_name"],
            event.payload["imagery_ion_id"],
        )

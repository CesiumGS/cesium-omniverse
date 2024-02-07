from __future__ import annotations
from typing import Optional
import carb.events


class AssetToAdd:
    def __init__(
        self,
        tileset_name: str,
        tileset_ion_asset_id: int,
        raster_overlay_name: Optional[str] = None,
        raster_overlay_ion_asset_id: Optional[int] = None,
    ):
        self.tileset_name = tileset_name
        self.tileset_ion_asset_id = tileset_ion_asset_id
        self.raster_overlay_name = raster_overlay_name
        self.raster_overlay_ion_asset_id = raster_overlay_ion_asset_id

    def to_dict(self) -> dict:
        return {
            "tileset_name": self.tileset_name,
            "tileset_ion_asset_id": self.tileset_ion_asset_id,
            "raster_overlay_name": self.raster_overlay_name,
            "raster_overlay_ion_asset_id": self.raster_overlay_ion_asset_id,
        }

    @staticmethod
    def from_event(event: carb.events.IEvent) -> Optional[AssetToAdd]:
        if event.payload is None or len(event.payload) == 0:
            return None

        return AssetToAdd(
            event.payload["tileset_name"],
            event.payload["tileset_ion_asset_id"],
            event.payload["raster_overlay_name"],
            event.payload["raster_overlay_ion_asset_id"],
        )

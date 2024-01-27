from __future__ import annotations
from typing import Optional
import carb.events


class RasterOverlayToAdd:
    def __init__(self, tileset_path: str, raster_overlay_ion_asset_id: int, raster_overlay_name: str):
        self.tileset_path = tileset_path
        self.raster_overlay_ion_asset_id = raster_overlay_ion_asset_id
        self.raster_overlay_name = raster_overlay_name

    def to_dict(self) -> dict:
        return {
            "tileset_path": self.tileset_path,
            "raster_overlay_ion_asset_id": self.raster_overlay_ion_asset_id,
            "raster_overlay_name": self.raster_overlay_name,
        }

    @staticmethod
    def from_event(event: carb.events.IEvent) -> Optional[RasterOverlayToAdd]:
        if event.payload is None or len(event.payload) == 0:
            return None

        return RasterOverlayToAdd(
            event.payload["tileset_path"],
            event.payload["raster_overlay_ion_asset_id"],
            event.payload["raster_overlay_name"],
        )

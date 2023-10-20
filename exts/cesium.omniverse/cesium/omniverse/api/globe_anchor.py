from pxr.Sdf import Path
from typing import Optional

from ..utils.cesium_interface import CesiumInterfaceManager


def anchor_xform_at_path(
    path: Path, latitude: Optional[float] = None, longitude: Optional[float] = None, height: Optional[float] = None
):
    with CesiumInterfaceManager() as interface:
        if latitude is None and longitude is None and height is None:
            interface.add_global_anchor_to_prim(str(path))
        else:
            interface.add_global_anchor_to_prim(str(path), latitude, longitude, height)

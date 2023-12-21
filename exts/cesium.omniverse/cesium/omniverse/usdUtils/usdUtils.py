import omni.usd
import omni.kit
import re
from typing import List

from cesium.usd.plugins.CesiumUsdSchemas import (
    IonImagery as CesiumIonImagery,
    Tileset as CesiumTileset,
    Tokens as CesiumTokens,
    Data as CesiumData,
    CartographicPolygon as CesiumCartographicPolygon,
)
from pxr import Sdf
from pxr.UsdGeom import Gprim

CESIUM_DATA_PRIM_PATH = "/Cesium"


def add_tileset_ion(name: str, tileset_id: int, token: str = "") -> str:
    stage = omni.usd.get_context().get_stage()

    safe_name = re.sub("[\\W]+", "_", name)
    if not safe_name.startswith("/"):
        safe_name = "/" + safe_name

    tileset_path: str = omni.usd.get_stage_next_free_path(stage, safe_name, False)
    tileset = CesiumTileset.Define(stage, tileset_path)
    assert tileset.GetPrim().IsValid()

    tileset.GetIonAssetIdAttr().Set(tileset_id)
    tileset.GetIonAccessTokenAttr().Set(token)
    tileset.GetSourceTypeAttr().Set(CesiumTokens.ion)

    server_prim_path = get_path_to_current_ion_server()
    if server_prim_path != Sdf.Path.emptyPath:
        tileset.GetIonServerBindingRel().AddTarget(server_prim_path)

    return tileset_path


def add_imagery_ion(tileset_path: str, name: str, asset_id: int, token: str = "") -> str:
    stage = omni.usd.get_context().get_stage()

    safe_name = re.sub("[\\W]+", "_", name)

    imagery_path: str = Sdf.Path(tileset_path).AppendPath(safe_name)

    # get_stage_next_free_path will increment the path name if there is a colllision
    imagery_path: str = Sdf.Path(omni.usd.get_stage_next_free_path(stage, imagery_path, False))

    imagery = CesiumIonImagery.Define(stage, imagery_path)
    assert imagery.GetPrim().IsValid()
    parent = imagery.GetPrim().GetParent()
    assert parent.IsA(CesiumTileset)

    imagery.GetIonAssetIdAttr().Set(asset_id)
    imagery.GetIonAccessTokenAttr().Set(token)

    server_prim_path = get_path_to_current_ion_server()
    if server_prim_path != Sdf.Path.emptyPath:
        imagery.GetIonServerBindingRel().AddTarget(server_prim_path)

    return imagery_path


def add_cartographic_polygon() -> None:
    stage = omni.usd.get_context().get_stage()
    # safe_name = "cartographic_polygon"  # TODO
    # cartographic_polygon_path: str = Sdf.Path("/CesiumCartographicPolygon").AppendPath(safe_name)
    # cartographic_polygon_path = Sdf.Path(omni.usd.get_stage_next_free_path(stage, cartographic_polygon_path, False))
    cartographic_polygon_path: str = Sdf.Path("/CesiumCartographicPolygon")

    cartographic_polygon = CesiumCartographicPolygon.Define(stage, cartographic_polygon_path)
    assert cartographic_polygon.GetPrim().IsValid()


def is_tileset(maybe_tileset: Gprim) -> bool:
    return maybe_tileset.isA(CesiumTileset)


def remove_tileset(tileset_path: str) -> None:
    stage = omni.usd.get_context().get_stage()

    stage.RemovePrim(Sdf.Path(tileset_path))


def get_path_to_current_ion_server() -> Sdf.Path:
    stage = omni.usd.get_context().get_stage()

    data: CesiumData = CesiumData.Get(stage, CESIUM_DATA_PRIM_PATH)

    if not data.GetPrim().IsValid():
        return Sdf.Path()

    rel = data.GetSelectedIonServerRel()
    targets = rel.GetForwardedTargets()

    return targets[0]


def set_path_to_current_ion_server(server_path: str):
    stage = omni.usd.get_context().get_stage()

    data = CesiumData.Get(stage, CESIUM_DATA_PRIM_PATH)

    if not data.GetPrim().IsValid():
        return

    rel = data.GetSelectedIonServerRel()
    rel.SetTargets([server_path])


def get_all_tileset_paths() -> List[str]:
    stage = omni.usd.get_context().get_stage()
    tileset_paths = [x.GetPath().pathString for x in stage.Traverse() if x.IsA(CesiumTileset)]
    return tileset_paths

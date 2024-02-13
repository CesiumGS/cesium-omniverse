import omni.usd
import re
from pxr import Sdf
from typing import List, Optional
from pxr import UsdGeom

from cesium.usd.plugins.CesiumUsdSchemas import (
    Data as CesiumData,
    Tileset as CesiumTileset,
    IonRasterOverlay as CesiumIonRasterOverlay,
    Georeference as CesiumGeoreference,
    GlobeAnchorAPI as CesiumGlobeAnchorAPI,
    Tokens as CesiumTokens,
)

CESIUM_DATA_PRIM_PATH = "/Cesium"
CESIUM_GEOREFERENCE_PRIM_PATH = "/CesiumGeoreference"


def get_safe_name(name: str) -> str:
    return re.sub("[\\W]+", "_", name)


def get_or_create_cesium_data() -> CesiumData:
    stage = omni.usd.get_context().get_stage()

    path = CESIUM_DATA_PRIM_PATH
    prim = stage.GetPrimAtPath(path)

    if prim.IsValid():
        return CesiumData.Get(stage, path)

    return CesiumData.Define(stage, path)


def get_or_create_cesium_georeference() -> CesiumGeoreference:
    stage = omni.usd.get_context().get_stage()

    georeference_paths = get_georeference_paths()

    if len(georeference_paths) < 1:
        return CesiumGeoreference.Define(stage, CESIUM_GEOREFERENCE_PRIM_PATH)

    return CesiumGeoreference.Get(stage, georeference_paths[0])


def add_tileset_ion(name: str, asset_id: int, token: str = "") -> str:
    stage = omni.usd.get_context().get_stage()

    safe_name = get_safe_name(name)
    if not safe_name.startswith("/"):
        safe_name = "/" + safe_name

    # get_stage_next_free_path will increment the path name if there is a collision
    tileset_path = omni.usd.get_stage_next_free_path(stage, safe_name, False)

    tileset = CesiumTileset.Define(stage, tileset_path)

    tileset.GetIonAssetIdAttr().Set(asset_id)
    tileset.GetIonAccessTokenAttr().Set(token)
    tileset.GetSourceTypeAttr().Set(CesiumTokens.ion)

    georeference = get_or_create_cesium_georeference()
    georeference_path = georeference.GetPath().pathString
    tileset.GetGeoreferenceBindingRel().AddTarget(georeference_path)

    server_prim_path = get_path_to_current_ion_server()
    if server_prim_path != "":
        tileset.GetIonServerBindingRel().AddTarget(server_prim_path)

    return tileset_path


def add_raster_overlay_ion(tileset_path: str, name: str, asset_id: int, token: str = "") -> str:
    stage = omni.usd.get_context().get_stage()

    safe_name = get_safe_name(name)

    raster_overlay_path = Sdf.Path(tileset_path).AppendPath(safe_name).pathString

    # get_stage_next_free_path will increment the path name if there is a collision
    raster_overlay_path = omni.usd.get_stage_next_free_path(stage, raster_overlay_path, False)

    raster_overlay = CesiumIonRasterOverlay.Define(stage, raster_overlay_path)

    tileset_prim = CesiumTileset.Get(stage, tileset_path)
    tileset_prim.GetRasterOverlayBindingRel().AddTarget(raster_overlay_path)

    raster_overlay.GetIonAssetIdAttr().Set(asset_id)
    raster_overlay.GetIonAccessTokenAttr().Set(token)

    server_prim_path = get_path_to_current_ion_server()
    if server_prim_path != "":
        raster_overlay.GetIonServerBindingRel().AddTarget(server_prim_path)

    return raster_overlay_path


def add_cartographic_polygon() -> str:
    stage = omni.usd.get_context().get_stage()

    name = "cartographic_polygon"
    cartographic_polygon_path = Sdf.Path("/CesiumCartographicPolygons").AppendPath(name).pathString
    cartographic_polygon_path = omni.usd.get_stage_next_free_path(stage, cartographic_polygon_path, False)

    basis_curves = UsdGeom.BasisCurves.Define(stage, cartographic_polygon_path)
    basis_curves.GetTypeAttr().Set("linear")
    basis_curves.GetWrapAttr().Set("periodic")

    # Set curve to have 10m edge lengths
    curve_size = 10 / UsdGeom.GetStageMetersPerUnit(stage)
    basis_curves.GetPointsAttr().Set(
        [
            (-curve_size, 0, -curve_size),
            (-curve_size, 0, curve_size),
            (curve_size, 0, curve_size),
            (curve_size, 0, -curve_size),
        ]
    )

    basis_curves.GetCurveVertexCountsAttr().Set([4])

    # Set curve to a 0.5m width
    curve_width = 0.5 / UsdGeom.GetStageMetersPerUnit(stage)
    basis_curves.GetWidthsAttr().Set([curve_width, curve_width, curve_width, curve_width])

    add_globe_anchor_to_prim(cartographic_polygon_path)

    return cartographic_polygon_path


def is_tileset(path: str) -> bool:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    return prim.IsA(CesiumTileset)


def remove_tileset(path: str) -> None:
    stage = omni.usd.get_context().get_stage()
    stage.RemovePrim(path)


def get_path_to_current_ion_server() -> Optional[str]:
    data = get_or_create_cesium_data()
    rel = data.GetSelectedIonServerRel()
    targets = rel.GetForwardedTargets()

    if len(targets) < 1:
        return None

    return targets[0].pathString


def set_path_to_current_ion_server(path: str) -> None:
    data = get_or_create_cesium_data()
    rel = data.GetSelectedIonServerRel()

    # This check helps avoid sending unnecessary USD notifications
    # See https://github.com/CesiumGS/cesium-omniverse/issues/640
    if get_path_to_current_ion_server() != path:
        rel.SetTargets([path])


def get_tileset_paths() -> List[str]:
    stage = omni.usd.get_context().get_stage()
    paths = [x.GetPath().pathString for x in stage.Traverse() if x.IsA(CesiumTileset)]
    return paths


def get_georeference_paths() -> List[str]:
    stage = omni.usd.get_context().get_stage()
    paths = [x.GetPath().pathString for x in stage.Traverse() if x.IsA(CesiumGeoreference)]
    return paths


def add_globe_anchor_to_prim(path: str) -> CesiumGlobeAnchorAPI:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    georeference_path = get_or_create_cesium_georeference().GetPath().pathString

    globe_anchor = CesiumGlobeAnchorAPI.Apply(prim)
    globe_anchor.GetGeoreferenceBindingRel().AddTarget(georeference_path)

    return globe_anchor

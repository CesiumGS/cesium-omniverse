from typing import Optional, Callable
import omni.usd
import omni.kit
import omni.ui as ui
import re
from cesium.usd.plugins.CesiumUsdSchemas import (
    Imagery as CesiumImagery,
    Tileset as CesiumTileset,
    Tokens as CesiumTokens,
)
from pxr import Sdf


async def wait_n_frames(n: int):
    for i in range(0, n):
        await omni.kit.app.get_app().next_update_async()


async def dock_window_async(
    window: Optional[ui.Window], target: str = "Stage", position: ui.DockPosition = ui.DockPosition.SAME
):
    if window is None:
        return

    # Wait five frame
    await wait_n_frames(5)
    stage_window = ui.Workspace.get_window(target)
    window.dock_in(stage_window, position, 1)
    window.focus()


async def perform_action_after_n_frames_async(n: int, action: Callable[[], None]):
    await wait_n_frames(n)
    action()


def str_is_empty_or_none(s: Optional[str]):
    if s is None:
        return True

    if s == "":
        return True

    return False


def add_tileset_ion(name: str, tilesetId: int, token: str = ""):
    stage = omni.usd.get_context().get_stage()

    safeName = re.sub("[\\W]+", "_", name)

    tileset_path = omni.usd.get_stage_next_free_path(stage, safeName, False)
    tileset = CesiumTileset.Define(stage, tileset_path)
    assert tileset.GetPrim().IsValid()

    tileset.GetIonAssetIdAttr().Set(tilesetId)
    tileset.GetIonAccessTokenAttr().Set(token)
    tileset.GetSourceTypeAttr().Set(CesiumTokens.ion)

    return tileset_path


def add_imagery_ion(tileset_path: str, name: str, asset_id: int, token: str = ""):
    stage = omni.usd.get_context().get_stage()

    safeName = re.sub("[\\W]+", "_", name)

    imagery_path = Sdf.Path(tileset_path).AppendPath(safeName)

    # get_stage_next_free_path will increment the path name if there is a colllision
    imagery_path = Sdf.Path(omni.usd.get_stage_next_free_path(stage, imagery_path, False))

    imagery = CesiumImagery.Define(stage, imagery_path)
    assert imagery.GetPrim().IsValid()
    parent = imagery.GetPrim().GetParent()
    assert parent.IsA(CesiumTileset)

    imagery.GetIonAssetIdAttr().Set(asset_id)
    imagery.GetIonAccessTokenAttr().Set(token)

    return imagery_path


def is_tileset(maybeTileset):
    return maybeTileset.isA(CesiumTileset)


def remove_tileset(tilesetPath: Sdf.Path):
    stage = omni.usd.get_context().get_stage()

    stage.RemovePrim(tilesetPath)

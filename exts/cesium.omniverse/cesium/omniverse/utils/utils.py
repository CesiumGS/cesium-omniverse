from typing import Optional, Callable
from pxr import Gf, UsdGeom
import omni.usd
import omni.kit
import omni.ui as ui
from omni.kit.viewport.utility import get_active_viewport


async def wait_n_frames(n: int) -> None:
    for i in range(0, n):
        await omni.kit.app.get_app().next_update_async()


async def dock_window_async(
    window: Optional[ui.Window], target: str = "Stage", position: ui.DockPosition = ui.DockPosition.SAME
) -> None:
    if window is None:
        return

    # Wait five frame
    await wait_n_frames(5)
    stage_window = ui.Workspace.get_window(target)
    window.dock_in(stage_window, position, 1)
    window.focus()


async def perform_action_after_n_frames_async(n: int, action: Callable[[], None]) -> None:
    await wait_n_frames(n)
    action()


def str_is_empty_or_none(s: Optional[str]) -> bool:
    if s is None:
        return True

    if s == "":
        return True

    return False


def extend_far_plane():
    stage = omni.usd.get_context().get_stage()
    viewport = get_active_viewport()
    camera_path = viewport.get_active_camera()
    camera = UsdGeom.Camera.Get(stage, camera_path)
    assert camera.GetPrim().IsValid()

    scoped_edit = ScopedEdit(stage)  # noqa: F841
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(1.0, 10000000000.0))

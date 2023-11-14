from typing import Optional, Callable
import omni.usd
import omni.kit
import omni.ui as ui


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

from typing import Optional
import omni.kit
import omni.ui as ui


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

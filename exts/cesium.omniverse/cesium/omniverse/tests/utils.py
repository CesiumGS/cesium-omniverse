from pathlib import Path
import omni.kit.app


def get_golden_img_dir():
    manager = omni.kit.app.get_app().get_extension_manager()
    ext_id = manager.get_extension_id_by_module("cesium.omniverse")
    return Path(manager.get_extension_path(ext_id)).joinpath("images/tests/ui/pass_fail_widget")


async def wait_for_update(wait_frames=10):
    for _ in range(wait_frames):
        await omni.kit.app.get_app().next_update_async()

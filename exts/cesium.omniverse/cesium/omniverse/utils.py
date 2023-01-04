import omni.kit


async def wait_n_frames(n: int):
    for i in range(0, n):
        await omni.kit.app.get_app().next_update_async()

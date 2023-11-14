import omni.ext
import carb.settings
import asyncio
from cesium.omniverse.utils import perform_action_after_n_frames_async
import os
from cesium.omniverse.utils.cesium_interface import CesiumInterfaceManager
import omni.usd
import omni.kit
import omni.kit.commands
# import cesium.


# Functions and vars are available to other extension as usual in python: `example.python_ext.some_public_function(x)`

# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class CesiumProfilingExtension(omni.ext.IExt):

    def __init__(self):
        super().__init__()

    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        print("cesium.profiling.app startup")

        settings = carb.settings.get_settings()
        if (settings.get("/app/runProfilingScenes")):
            # wait several seconds for Omniverse to load before running the first test
            asyncio.ensure_future(perform_action_after_n_frames_async(120, self._run_profiling_suite))

    def on_shutdown(self):
        print("[company.hello.world] company hello world shutdown")

    def _run_profiling_suite(self):
        asyncio.ensure_future(self._run_profiling_suite_async())

    def _stop_profiler(self):
        with CesiumInterfaceManager() as interface:
            interface.shut_down_profiling()

    def _start_profiler(self, file_base_name):
        with CesiumInterfaceManager() as interface:
            print(f"Initializing profiling at {file_base_name}")
            interface.initialize_profiling(file_base_name)

    def _get_profiling_files(self, directory_path):
        files = []
        files_in_directory = os.listdir(directory_path)
        for file_name in files_in_directory:
            if file_name.endswith('.usdc'):
                usd_file_path = os.path.join(directory_path, file_name)
                files.append(usd_file_path)
        return files

    async def _run_profiling_suite_async(self):
        scene_test_duration = 20
        between_test_scene_duration = 5
        current_working_directory = os.getcwd()
        test_directory = os.path.join(current_working_directory, "tests/testAssets/usd/flattened")
        profiling_usd_files = self._get_profiling_files(test_directory)

        for usd_file in profiling_usd_files:
            file_base_name = os.path.splitext(os.path.basename(usd_file))[0]
            stage = omni.usd.get_context().open_stage(usd_file)
            # extend_far_plane()
            if stage:
                self._start_profiler(file_base_name)
                omni.kit.commands.execute('ToolbarPlayButtonClicked')
                await asyncio.sleep(scene_test_duration)
                omni.kit.commands.execute('ToolbarStopButtonClicked')
                omni.usd.get_context().close_stage()
                self._stop_profiler()
                await asyncio.sleep(between_test_scene_duration)
            else:
                print(f"Could not open file {usd_file}")

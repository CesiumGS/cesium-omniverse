import carb.settings
import asyncio
from cesium.omniverse.utils import perform_action_after_n_frames_async
from cesium.omniverse.utils.cesium_interface import CesiumInterfaceManager
import omni.ext
import omni.usd
import omni.kit
import omni.kit.commands
import logging
import os


class CesiumProfilingExtension(omni.ext.IExt):

    def __init__(self):
        super().__init__()

        self._logger: logging.Logger = logging.getLogger(__name__)

    def on_startup(self, ext_id):
        settings = carb.settings.get_settings()
        if (settings.get("/app/runProfilingScenes")):
            FRAMES_BEFORE_TESTING_BEGINS = 120
            asyncio.ensure_future(perform_action_after_n_frames_async(FRAMES_BEFORE_TESTING_BEGINS,
                                                                      self._run_profiling_suite))

    def _run_profiling_suite(self):
        asyncio.ensure_future(self._run_profiling_suite_async())

    def _stop_profiler(self):
        with CesiumInterfaceManager() as interface:
            interface.shut_down_profiling()

    def _start_profiler(self, file_basename):
        with CesiumInterfaceManager() as interface:
            interface.initialize_profiling(file_basename)

    def _get_profiling_files(self, directory_path):
        files = []
        files_in_directory = os.listdir(directory_path)
        for file_name in files_in_directory:
            if file_name.endswith('.usdc'):
                usd_file_path = os.path.join(directory_path, file_name)
                files.append(usd_file_path)
        return files

    async def _run_profiling_suite_async(self):
        self._stop_profiler()

        scene_test_duration = 20
        between_test_scene_duration = 5
        current_working_directory = os.getcwd()
        test_directory = os.path.join(current_working_directory, "tests/testAssets/usd/flattened")
        profiling_usd_files = self._get_profiling_files(test_directory)

        for usd_file in profiling_usd_files:
            file_basename = os.path.splitext(os.path.basename(usd_file))[0]
            stage = omni.usd.get_context().open_stage(usd_file)
            if stage:
                self._start_profiler(file_basename)
                omni.kit.commands.execute('ToolbarPlayButtonClicked')
                await asyncio.sleep(scene_test_duration)
                omni.kit.commands.execute('ToolbarStopButtonClicked')
                omni.usd.get_context().close_stage()
                self._stop_profiler()
                await asyncio.sleep(between_test_scene_duration)
            else:
                self._logger.warning("Could not open file" + usd_file)

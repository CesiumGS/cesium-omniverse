import os
import omni.ext
import omni.usd
import omni.kit.ui
from .bindings import acquire_cesium_omniverse_tests_interface, release_cesium_omniverse_tests_interface


class CesiumOmniverseCppTestsExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()

    def on_startup(self):
        print("Starting Cesium Tests Extension...")

        global tests_interface
        tests_interface = acquire_cesium_omniverse_tests_interface()

        tests_interface.on_startup(os.path.join(os.path.dirname(__file__), "../../../../../cesium.omniverse"))
        # tests_interface.on_startup("/home/melser/git/cesium-omniverse/exts/cesium.omniverse/cesium/omniverse/../../")

        # TODO ensure the stage has been set up before getting stage id
        stageId = omni.usd.get_context().get_stage_id()

        tests_interface.run_all_tests(stageId)

        print("Started Cesium Tests Extension.")

    def on_shutdown(self):
        print("Stopping Cesium Tests Extension...")
        release_cesium_omniverse_tests_interface(tests_interface)
        print("Stopped Cesium Tests Extension.")

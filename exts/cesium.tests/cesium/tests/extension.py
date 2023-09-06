import omni.ext
import omni.kit.ui
from .bindings import acquire_cesium_omniverse_tests_interface, release_cesium_omniverse_tests_interface


class CesiumTestsExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()

    def on_startup(self):
        print("Starting Cesium Tests Extension...")

        global tests_interface
        tests_interface = acquire_cesium_omniverse_tests_interface()

        tests_interface.run_all_tests(0)

        print("Started Cesium Tests Extension.")

    def on_shutdown(self):
        print("Stopping Cesium Tests Extension...")
        release_cesium_omniverse_tests_interface(tests_interface)
        print("Stopped Cesium Tests Extension.")

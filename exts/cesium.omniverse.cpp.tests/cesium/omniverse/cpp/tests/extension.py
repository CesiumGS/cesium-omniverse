import os
import omni.ext
import omni.usd
import omni.kit.ui
import omni.kit.app
from .bindings import acquire_cesium_omniverse_tests_interface, release_cesium_omniverse_tests_interface


class CesiumOmniverseCppTestsExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()

    def on_startup(self):
        print("Starting Cesium Tests Extension...")

        global tests_interface
        tests_interface = acquire_cesium_omniverse_tests_interface()

        tests_interface.on_startup(os.path.join(os.path.dirname(__file__), "../../../../../cesium.omniverse"))

        # update_stream = omni.kit.app.get_app().get_update_event_stream()

        # To ensure the tests only run after the stage has been opened, we
        # attach a handler to an event that occurs every frame. That handler
        # checks if the stage has opened, runs once, then detaches itself
        # self._run_once_sub = update_stream.create_subscription_to_pop(
        #     self.run_once_after_stage_opens, name="Run once after stage opens"
        # )

        print("Started Cesium Tests Extension.")

    # def run_once_after_stage_opens(self, _):
    #     if omni.usd.get_context().get_stage_state() == omni.usd.StageState.OPENED:
    #         print("Beginning Cesium Tests Extension tests")
    #         stageId = omni.usd.get_context().get_stage_id()
    #         tests_interface.run_all_tests(stageId)
    #         print("Cesium Tests Extension tests complete")
    #         self._run_once_sub.unsubscribe()

    def on_shutdown(self):
        print("Stopping Cesium Tests Extension...")
        tests_interface.on_shutdown()
        release_cesium_omniverse_tests_interface(tests_interface)
        print("Stopped Cesium Tests Extension.")

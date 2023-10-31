import os
import omni.ext
import omni.usd
import omni.kit.ui
import omni.kit.app
from .bindings import acquire_cesium_omniverse_tests_interface, release_cesium_omniverse_tests_interface


class CesiumOmniverseCppTestsExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()
        self.tests_set_up = False

    def on_startup(self):
        print("Starting Cesium Tests Extension...")

        global tests_interface
        tests_interface = acquire_cesium_omniverse_tests_interface()

        tests_interface.on_startup(os.path.join(os.path.dirname(__file__), "../../../../../cesium.omniverse"))

        update_stream = omni.kit.app.get_app().get_update_event_stream()

        # To ensure the tests only run after the stage has been opened, we
        # attach a handler to an event that occurs every frame. That handler
        # checks if the stage has opened, runs once, then detaches itself
        self._run_once_sub = update_stream.create_subscription_to_pop(
            self.run_once_after_stage_opens, name="Run once after stage opens"
        )

        print("Started Cesium Tests Extension.")

    def run_once_after_stage_opens(self, _):
        # wait until the USD stage is fully set up
        if omni.usd.get_context().get_stage_state() == omni.usd.StageState.OPENED:
            # set up tests on one frame, then run the tests on the next frame
            # note we can't use wait_n_frames here as this is a subscribed function
            # so it cannot be async
            if not self.tests_set_up:
                self.tests_set_up = True
                print("Beginning Cesium Tests Extension tests")
                stageId = omni.usd.get_context().get_stage_id()
                tests_interface.set_up_tests(stageId)
            else:
                # unsubscribe so there's no way the next frame triggers another run
                self._run_once_sub.unsubscribe()
                tests_interface.run_all_tests()
                print("Cesium Tests Extension tests complete")

    def on_shutdown(self):
        print("Stopping Cesium Tests Extension...")
        tests_interface.on_shutdown()
        release_cesium_omniverse_tests_interface(tests_interface)
        print("Stopped Cesium Tests Extension.")

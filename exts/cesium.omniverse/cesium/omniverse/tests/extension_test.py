import omni.kit.test
import omni.kit.ui_test as ui_test
from typing import Optional


_window_ref: Optional[ui_test.WidgetRef] = None


class ExtensionTest(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        global _window_ref
        _window_ref = ui_test.find("Cesium")

    async def tearDown(self):
        pass

    async def test_cesium_window_opens(self):
        global _window_ref
        self.assertIsNotNone(_window_ref)

    async def test_window_docked(self):
        await ui_test.wait_n_updates(4)
        global _window_ref
        self.assertTrue(_window_ref.window.docked)

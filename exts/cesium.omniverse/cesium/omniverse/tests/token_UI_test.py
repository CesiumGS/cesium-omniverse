import omni.kit.test
import omni.kit.ui_test as ui_test
from typing import Optional

_CesiumWindow: Optional[ui_test.WidgetRef] = None


class TokenUITest(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        # global _CesiumWindow
        # _CesiumWindow = ui_test.find("Cesium")

        tokenButton = ui_test.find("*Cesium/Token")

        await tokenButton.click()

        # await _CesiumWindow.click(ui_test.Vec2(160, 75))
        # await _CesiumWindow.click(ui_test.Vec2(740, 333))

        global _TokenWindow
        _TokenWindow = ui_test.find("*Select Cesium ion Token")

    # async def tearDown(self):
    #     pass

    async def test_token_window_opens(self):
        global _TokenWindow
        self.assertIsNotNone(_TokenWindow)

    # async def test_window_docked(self):
    #     await ui_test.wait_n_updates(4)
    #     global _window_ref
    #     self.assertTrue(_window_ref.window.docked)

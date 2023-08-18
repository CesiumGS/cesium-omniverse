import omni.kit.test
import omni.kit.ui_test as ui_test
import omni.usd
import pxr.Usd

import cesium.usd
from typing import Optional


_window_ref: Optional[ui_test.WidgetRef] = None


class ExtensionTest(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        global _window_ref

        # can be removed (or at least decreased) once there is no delay
        # required before spawning the cesium window. See:
        # https://github.com/CesiumGS/cesium-omniverse/pull/423
        await ui_test.wait_n_updates(24)

        _window_ref = ui_test.find("Cesium")

    async def tearDown(self):
        pass

    async def test_cesium_window_opens(self):
        global _window_ref
        self.assertIsNotNone(_window_ref)

    async def test_window_docked(self):
        global _window_ref
        # docked is false if the window is not in focus,
        # as may be the case if other extensions are loaded
        await _window_ref.focus()
        self.assertTrue(_window_ref.window.docked)

    async def test_blank_tileset(self):
        global _window_ref

        blankTilesetButton = _window_ref.find("**/Button[*].text=='Blank 3D Tiles Tileset'")
        self.assertIsNotNone(blankTilesetButton)

        stage: pxr.Usd.Stage = omni.usd.get_context().get_stage()
        self.assertIsNotNone(stage)

        self.assertFalse(any([i.IsA(cesium.usd.plugins.CesiumUsdSchemas.Tileset) for i in stage.Traverse()]))

        await blankTilesetButton.click()

        await ui_test.wait_n_updates(2)  # passes without, but seems prudent
        self.assertTrue(any([i.IsA(cesium.usd.plugins.CesiumUsdSchemas.Tileset) for i in stage.Traverse()]))

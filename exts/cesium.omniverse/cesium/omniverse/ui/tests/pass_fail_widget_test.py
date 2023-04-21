import omni.ui as ui
from omni.ui.tests.test_base import OmniUiTest
from cesium.omniverse.tests.utils import get_golden_img_dir, wait_for_update
from cesium.omniverse.ui.pass_fail_widget import CesiumPassFailWidget


class PassFailWidgetTest(OmniUiTest):
    async def setUp(self):
        await super().setUp()
        self._golden_image_dir = get_golden_img_dir()

    async def tearDown(self):
        await super().tearDown()

    async def test_pass_fail_widget_passed(self):
        window = await self.create_test_window()

        with window.frame:
            with ui.VStack(height=0):
                widget = CesiumPassFailWidget(True)

                self.assertIsNotNone(widget)
                self.assertTrue(widget.passed)

        await wait_for_update()

        await self.finalize_test(
            golden_img_dir=self._golden_image_dir, golden_img_name="test_pass_fail_widget_passed.png"
        )

        widget.destroy()

    async def test_pass_fail_widget_failed(self):
        window = await self.create_test_window()

        with window.frame:
            with ui.VStack(height=0):
                widget = CesiumPassFailWidget(False)

                self.assertIsNotNone(widget)
                self.assertFalse(widget.passed)

        await wait_for_update()

        await self.finalize_test(
            golden_img_dir=self._golden_image_dir, golden_img_name="test_pass_fail_widget_failed.png"
        )

        widget.destroy()

    async def test_pass_fail_widget_updated(self):
        window = await self.create_test_window()

        with window.frame:
            with ui.VStack(height=0):
                widget = CesiumPassFailWidget(False)

                self.assertIsNotNone(widget)
                self.assertFalse(widget.passed)

        await wait_for_update()

        widget.passed = True

        await wait_for_update()

        self.assertTrue(widget.passed)

        await self.finalize_test(
            golden_img_dir=self._golden_image_dir, golden_img_name="test_pass_fail_widget_changed.png"
        )

        widget.destroy()

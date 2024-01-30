import carb.events
import omni.kit.test
from unittest.mock import MagicMock
from cesium.omniverse.models.raster_overlay_to_add import RasterOverlayToAdd

TILESET_PATH = "/fake/tileset/path"
RASTER_OVERLAY_NAME = "fake_raster_overlay_name"
RASTER_OVERLAY_ION_ASSET_ID = 2
PAYLOAD_DICT = {
    "tileset_path": TILESET_PATH,
    "raster_overlay_name": RASTER_OVERLAY_NAME,
    "raster_overlay_ion_asset_id": RASTER_OVERLAY_ION_ASSET_ID,
}


class RasterOverlayToAddTest(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        pass

    async def tearDown(self):
        pass

    async def test_convert_raster_overlay_to_add_to_dict(self):
        raster_overlay_to_add = RasterOverlayToAdd(
            tileset_path=TILESET_PATH,
            raster_overlay_ion_asset_id=RASTER_OVERLAY_ION_ASSET_ID,
            raster_overlay_name=RASTER_OVERLAY_NAME,
        )

        result = raster_overlay_to_add.to_dict()
        self.assertEqual(result["tileset_path"], TILESET_PATH)
        self.assertEqual(result["raster_overlay_name"], RASTER_OVERLAY_NAME)
        self.assertEqual(result["raster_overlay_ion_asset_id"], RASTER_OVERLAY_ION_ASSET_ID)

    async def test_create_raster_overlay_to_add_from_event(self):
        mock_event = MagicMock(spec=carb.events.IEvent)
        mock_event.payload = PAYLOAD_DICT
        raster_overlay_to_add = RasterOverlayToAdd.from_event(mock_event)
        self.assertIsNotNone(raster_overlay_to_add)
        self.assertEqual(raster_overlay_to_add.tileset_path, TILESET_PATH)
        self.assertEqual(raster_overlay_to_add.raster_overlay_name, RASTER_OVERLAY_NAME)
        self.assertEqual(raster_overlay_to_add.raster_overlay_ion_asset_id, RASTER_OVERLAY_ION_ASSET_ID)

    async def test_create_raster_overlay_to_add_from_empty_event(self):
        mock_event = MagicMock(spec=carb.events.IEvent)
        mock_event.payload = None
        raster_overlay_to_add = RasterOverlayToAdd.from_event(mock_event)
        self.assertIsNone(raster_overlay_to_add)

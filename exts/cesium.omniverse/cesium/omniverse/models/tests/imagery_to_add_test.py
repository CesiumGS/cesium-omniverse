import carb.events
import omni.kit.test
from unittest.mock import MagicMock
from cesium.omniverse.models.imagery_to_add import ImageryToAdd

TILESET_PATH = "/fake/tileset/path"
IMAGERY_NAME = "fake_imagery_name"
IMAGERY_ION_ASSET_ID = 2
PAYLOAD_DICT = {
    "tileset_path": TILESET_PATH,
    "imagery_name": IMAGERY_NAME,
    "imagery_ion_asset_id": IMAGERY_ION_ASSET_ID,
}


class ImageryToAddTest(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        pass

    async def tearDown(self):
        pass

    async def test_convert_imagery_to_add_to_dict(self):
        imagery_to_add = ImageryToAdd(
            tileset_path=TILESET_PATH,
            imagery_ion_asset_id=IMAGERY_ION_ASSET_ID,
            imagery_name=IMAGERY_NAME,
        )

        result = imagery_to_add.to_dict()
        self.assertEqual(result["tileset_path"], TILESET_PATH)
        self.assertEqual(result["imagery_name"], IMAGERY_NAME)
        self.assertEqual(result["imagery_ion_asset_id"], IMAGERY_ION_ASSET_ID)

    async def test_create_imagery_to_add_from_event(self):
        mock_event = MagicMock(spec=carb.events.IEvent)
        mock_event.payload = PAYLOAD_DICT
        imagery_to_add = ImageryToAdd.from_event(mock_event)
        self.assertIsNotNone(imagery_to_add)
        self.assertEqual(imagery_to_add.tileset_path, TILESET_PATH)
        self.assertEqual(imagery_to_add.imagery_name, IMAGERY_NAME)
        self.assertEqual(imagery_to_add.imagery_ion_asset_id, IMAGERY_ION_ASSET_ID)

    async def test_create_imagery_to_add_from_empty_event(self):
        mock_event = MagicMock(spec=carb.events.IEvent)
        mock_event.payload = None
        imagery_to_add = ImageryToAdd.from_event(mock_event)
        self.assertIsNone(imagery_to_add)

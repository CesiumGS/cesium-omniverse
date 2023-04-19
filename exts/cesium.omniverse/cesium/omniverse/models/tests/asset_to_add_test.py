import carb.events
import omni.kit.test
from unittest.mock import MagicMock
from cesium.omniverse.models.asset_to_add import AssetToAdd

TILESET_NAME = "fake_tileset_name"
TILESET_ION_ASSET_ID = 1
IMAGERY_NAME = "fake_imagery_name"
IMAGERY_ION_ASSET_ID = 2
PAYLOAD_DICT = {
    "tileset_name": TILESET_NAME,
    "tileset_ion_asset_id": TILESET_ION_ASSET_ID,
    "imagery_name": IMAGERY_NAME,
    "imagery_ion_asset_id": IMAGERY_ION_ASSET_ID,
}


class AssetToAddTest(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        pass

    async def tearDown(self):
        pass

    async def test_convert_asset_to_add_to_dict(self):
        asset_to_add = AssetToAdd(
            tileset_name=TILESET_NAME,
            tileset_ion_asset_id=TILESET_ION_ASSET_ID,
            imagery_name=IMAGERY_NAME,
            imagery_ion_asset_id=IMAGERY_ION_ASSET_ID,
        )

        result = asset_to_add.to_dict()
        self.assertEqual(result["tileset_name"], TILESET_NAME)
        self.assertEqual(result["tileset_ion_asset_id"], TILESET_ION_ASSET_ID)
        self.assertEqual(result["imagery_name"], IMAGERY_NAME)
        self.assertEqual(result["imagery_ion_asset_id"], IMAGERY_ION_ASSET_ID)

    async def test_create_asset_to_add_from_event(self):
        mock_event = MagicMock(spec=carb.events.IEvent)
        mock_event.payload = PAYLOAD_DICT
        asset_to_add = AssetToAdd.from_event(mock_event)
        self.assertIsNotNone(asset_to_add)
        self.assertEqual(asset_to_add.tileset_name, TILESET_NAME)
        self.assertEqual(asset_to_add.tileset_ion_asset_id, TILESET_ION_ASSET_ID)
        self.assertEqual(asset_to_add.imagery_name, IMAGERY_NAME)
        self.assertEqual(asset_to_add.imagery_ion_asset_id, IMAGERY_ION_ASSET_ID)

    async def test_create_asset_to_add_from_empty_event(self):
        mock_event = MagicMock(spec=carb.events.IEvent)
        mock_event.payload = None
        asset_to_add = AssetToAdd.from_event(mock_event)
        self.assertIsNone(asset_to_add)

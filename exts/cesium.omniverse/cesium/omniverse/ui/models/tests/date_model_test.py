import omni.kit.test
from cesium.omniverse.ui.models.date_model import DateModel


class Test(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        pass

    async def tearDown(self):
        pass

    async def test_date_model(self):
        input_value = "2016-10-17T22:04:30.353Z"
        expected_output = "2016-10-17"

        date_model = DateModel(input_value)
        self.assertEqual(date_model.as_string, expected_output)

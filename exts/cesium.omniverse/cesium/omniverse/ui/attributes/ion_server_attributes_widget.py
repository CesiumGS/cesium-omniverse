import logging
from omni.kit.property.usd.custom_layout_helper import CustomLayoutFrame, CustomLayoutGroup, CustomLayoutProperty
from omni.kit.property.usd.usd_property_widget import SchemaPropertiesWidget
from ...bindings import ICesiumOmniverseInterface
from cesium.usd.plugins.CesiumUsdSchemas import IonServer as CesiumIonServer


class CesiumIonServerAttributesWidget(SchemaPropertiesWidget):
    def __init__(self, _cesium_omniverse_interface: ICesiumOmniverseInterface):
        super().__init__("Cesium ion Server Settings", CesiumIonServer, include_inherited=False)

        self._logger = logging.getLogger(__name__)

        self._cesium_omniverse_interface = _cesium_omniverse_interface

    def clean(self):
        super().clean()

    def _customize_props_layout(self, props):
        frame = CustomLayoutFrame(hide_extra=True)

        with frame:
            with CustomLayoutGroup("ion Server"):
                CustomLayoutProperty("cesium:displayName")
                CustomLayoutProperty("cesium:ionServerUrl")
                CustomLayoutProperty("cesium:ionServerApiUrl")
                CustomLayoutProperty("cesium:ionServerApplicationId")
            with CustomLayoutGroup("Project Default Token"):
                CustomLayoutProperty("cesium:projectDefaultIonAccessToken")
                CustomLayoutProperty("cesium:projectDefaultIonAccessTokenId")

        return frame.apply(props)

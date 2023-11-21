import logging
import omni.ui as ui
import omni.usd
from .proj import epsg_to_ecef, epsg_to_wgs84, get_crs_name_from_epsg
import math
from .custom_fields import string_field_with_label, int_field_with_label, float_field_with_label
from pxr import Sdf


class CesiumGeorefHelperWindow(ui.Window):
    WINDOW_NAME = "Cesium Georeference Helper"

    _logger: logging.Logger

    def __init__(self, **kwargs):
        super().__init__(CesiumGeorefHelperWindow.WINDOW_NAME, **kwargs)

        self._logger = logging.getLogger(__name__)

        # Set the function that is called to build widgets when the window is visible
        self.frame.set_build_fn(self._build_fn)

    def destroy(self):
        # It will destroy all the children
        super().destroy()

    def __del__(self):
        self.destroy()

    @staticmethod
    def create_window():
        return CesiumGeorefHelperWindow(width=250, height=600)

    def _convert_coordinates(self):
        # Get the CRS and check if it is valid, adjust UI values accordingly
        crs = get_crs_name_from_epsg(self._epsg_model.get_value_as_int())
        if crs is None:
            self._epsg_name_model.set_value("Invalid EPSG Code")
            self._wgs84_latitude_model.set_value(math.nan)
            self._wgs84_longitude_model.set_value(math.nan)
            self._wgs84_height_model.set_value(math.nan)
            self._ecef_x_model.set_value(math.nan)
            self._ecef_y_model.set_value(math.nan)
            self._ecef_z_model.set_value(math.nan)
            return

        self._epsg_name_model.set_value(crs)

        # Convert coords to WGS84 and set in UI
        wgs84_coords = epsg_to_wgs84(
            self._epsg_model.as_string,
            self._easting_model.as_float,
            self._northing_model.as_float,
            self._elevation_model.as_float,
        )

        self._wgs84_latitude_model.set_value(wgs84_coords[0])
        self._wgs84_longitude_model.set_value(wgs84_coords[1])
        self._wgs84_height_model.set_value(wgs84_coords[2])

        # Convert coords to ECEF and set in UI
        ecef_coords = epsg_to_ecef(
            self._epsg_model.as_string,
            self._easting_model.as_float,
            self._northing_model.as_float,
            self._elevation_model.as_float,
        )

        self._ecef_x_model.set_value(ecef_coords[0])
        self._ecef_y_model.set_value(ecef_coords[1])
        self._ecef_z_model.set_value(ecef_coords[2])

    def _set_georeference_prim(self):
        if math.isnan(self._wgs84_latitude_model.get_value_as_float()):
            self._logger.warning("Cannot set CesiumGeoreference to NaN")
            return

        stage = omni.usd.get_context().get_stage()
        cesium_prim = stage.GetPrimAtPath("/CesiumGeoreference")
        cesium_prim.GetAttribute("cesium:georeferenceOrigin:latitude").Set(
            self._wgs84_latitude_model.get_value_as_float()
        )
        cesium_prim.GetAttribute("cesium:georeferenceOrigin:longitude").Set(
            self._wgs84_longitude_model.get_value_as_float()
        )
        cesium_prim.GetAttribute("cesium:georeferenceOrigin:height").Set(
            self._wgs84_height_model.get_value_as_float()
        )

    @staticmethod
    def set_georef_from_environment():
        stage = omni.usd.get_context().get_stage()

        environment_prim = stage.GetPrimAtPath("/Environment")
        cesium_prim = stage.GetPrimAtPath("/CesiumGeoreference")

        lat_attr = environment_prim.GetAttribute("location:latitude")
        long_attr = environment_prim.GetAttribute("location:longitude")

        if lat_attr and long_attr:
            cesium_prim.GetAttribute("cesium:georeferenceOrigin:latitude").Set(lat_attr.Get())
            cesium_prim.GetAttribute("cesium:georeferenceOrigin:longitude").Set(long_attr.Get())
            cesium_prim.GetAttribute("cesium:georeferenceOrigin:height").Set(0.0)
        else:
            logger = logging.getLogger(__name__)
            logger.warning(
                "Cannot set CesiumGeoreference as environment prim does not have latitude or longitude attributes"
            )

    @staticmethod
    def set_georef_from_anchor():
        logger = logging.getLogger(__name__)
        stage = omni.usd.get_context().get_stage()

        cesium_prim = stage.GetPrimAtPath("/CesiumGeoreference")

        if not cesium_prim.IsValid():
            logger.error("No CesiumGeoreference found")
            return

        selection = omni.usd.get_context().get_selection().get_selected_prim_paths()

        for prim_path in selection:
            prim = stage.GetPrimAtPath(prim_path)
            print("hello")
            coords = prim.GetAttribute("cesium:anchor:geographicCoordinates").Get()
            if coords is not None:
                print(coords)

                cesium_prim.GetAttribute("cesium:georeferenceOrigin:latitude").Set(coords[0])
                cesium_prim.GetAttribute("cesium:georeferenceOrigin:longitude").Set(coords[1])
                cesium_prim.GetAttribute("cesium:georeferenceOrigin:height").Set(coords[2])

                return

        logger.error("Please select a prim with a globe anchor")

    def _build_fn(self):
        """Builds out the UI buttons and their handlers."""

        with ui.VStack(spacing=4):
            label_style = {"Label": {"font_size": 16}}

            ui.Label(
                "Enter coordinates in any EPSG CRS to convert them to ECEF and WGS84",
                word_wrap=True,
                style=label_style,
            )
            ui.Spacer(height=10)
            ui.Label("Your Project Details:", style=label_style)

            # TODO: Precision issues to resolve
            def on_coordinate_update(event):
                self._convert_coordinates()

            # Define the SimpleValueModels for the UI
            self._epsg_model = ui.SimpleIntModel(28356)
            self._epsg_name_model = ui.SimpleStringModel("")
            self._easting_model = ui.SimpleFloatModel(503000.0)
            self._northing_model = ui.SimpleFloatModel(6950000.0)
            self._elevation_model = ui.SimpleFloatModel(0.0)

            # Add the value changed callbacks
            self._epsg_model.add_value_changed_fn(on_coordinate_update)
            self._easting_model.add_value_changed_fn(on_coordinate_update)
            self._northing_model.add_value_changed_fn(on_coordinate_update)
            self._elevation_model.add_value_changed_fn(on_coordinate_update)

            # TODO: Make EPSG an autocomplete field

            int_field_with_label("EPSG Code", model=self._epsg_model)
            string_field_with_label("EPSG Name", model=self._epsg_name_model, enabled=False)
            float_field_with_label("Easting / X", model=self._easting_model)
            float_field_with_label("Northing / Y", model=self._northing_model)
            float_field_with_label("Elevation / Z", model=self._elevation_model)

            ui.Spacer(height=10)

            # TODO: It would be nice to be able to copy these fields, or potentially have two way editing

            ui.Label("WGS84 Results:", style=label_style)
            self._wgs84_latitude_model = ui.SimpleFloatModel(0.0)
            self._wgs84_longitude_model = ui.SimpleFloatModel(0.0)
            self._wgs84_height_model = ui.SimpleFloatModel(0.0)
            float_field_with_label("Latitude", model=self._wgs84_latitude_model, enabled=False)
            float_field_with_label("Longitude", model=self._wgs84_longitude_model, enabled=False)
            float_field_with_label("Elevation", model=self._wgs84_height_model, enabled=False)

            ui.Spacer(height=10)

            ui.Label("ECEF Results:", style=label_style)
            self._ecef_x_model = ui.SimpleFloatModel(0.0)
            self._ecef_y_model = ui.SimpleFloatModel(0.0)
            self._ecef_z_model = ui.SimpleFloatModel(0.0)
            float_field_with_label("X", model=self._ecef_x_model, enabled=False)
            float_field_with_label("Y", model=self._ecef_y_model, enabled=False)
            float_field_with_label("Z", model=self._ecef_z_model, enabled=False)

            ui.Button("Set Georeference from EPSG", height=20, clicked_fn=self._set_georeference_prim)
            ui.Button(
                "Set Georeference from Environment Prim", height=20, clicked_fn=self.set_georef_from_environment
            )
            ui.Button("Set Georef from Selected Anchor", height=20, clicked_fn=self.set_georef_from_anchor)

            # Do the first conversion
            self._convert_coordinates()

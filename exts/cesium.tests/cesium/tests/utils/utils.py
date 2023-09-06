import omni.usd
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, UsdGeom, Sdf
import json
import carb.settings
import os


# Modified version of ScopedEdit in _build_viewport_cameras in omni.kit.widget.viewport
class ScopedEdit:
    def __init__(self, stage):
        self.__stage = stage
        edit_target = stage.GetEditTarget()
        edit_layer = edit_target.GetLayer()
        self.__edit_layer = stage.GetSessionLayer()
        self.__was_editable = self.__edit_layer.permissionToEdit
        if not self.__was_editable:
            self.__edit_layer.SetPermissionToEdit(True)
        if self.__edit_layer != edit_layer:
            stage.SetEditTarget(self.__edit_layer)
            self.__edit_target = edit_target
        else:
            self.__edit_target = None

    def __del__(self):
        if self.__edit_layer and not self.__was_editable:
            self.__edit_layer.SetPermissionToEdit(False)
            self.__edit_layer = None

        if self.__edit_target:
            self.__stage.SetEditTarget(self.__edit_target)
            self.__edit_target = None


def extend_far_plane():
    stage = omni.usd.get_context().get_stage()
    viewport = get_active_viewport()
    camera_path = viewport.get_active_camera()
    camera = UsdGeom.Camera.Get(stage, camera_path)
    assert camera.GetPrim().IsValid()

    scoped_edit = ScopedEdit(stage)  # noqa: F841
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(1.0, 10000000000.0))


def save_carb_settings(powertools_extension_location: str):
    carb_settings_path = os.path.join(powertools_extension_location, "carb_settings.txt")
    with open(carb_settings_path, "w") as fh:
        fh.write(json.dumps(carb.settings.get_settings().get("/"), indent=2))


# Helper function to search for an attribute on a prim, or create it if not present
def get_or_create_attribute(prim, name, type):
    attribute = prim.GetAttribute(name)
    if not attribute:
        attribute = prim.CreateAttribute(name, type)
    return attribute

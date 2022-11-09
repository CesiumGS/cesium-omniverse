#include "pyboost11.h"

#include "cesium/omniverse/CesiumOmniverse.h"

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec4d.h>
#include <pxr/usd/usd/stage.h>
#include <pybind11/pybind11.h>

namespace pybind11 {
namespace detail {

PYBOOST11_TYPE_CASTER(pxr::GfVec4d, _("Vec4d"));
PYBOOST11_TYPE_CASTER(pxr::GfMatrix4d, _("Matrix4d"));
PYBOOST11_TYPE_CASTER(pxr::UsdStageRefPtr, _("StageRefPtr"));

} // end namespace detail
} // end namespace pybind11

namespace Cesium {

PYBIND11_MODULE(CesiumOmniversePythonBindings, m) {
    m.doc() = "pybind11 cesium plugin"; // optional module docstring

    m.def("initialize", &initialize);
    m.def("finalize", &finalize);
    m.def("addTilesetUrl", &addTilesetUrl);
    m.def("addTilesetIon", &addTilesetIon);
    m.def("removeTileset", &removeTileset);
    m.def(
        "updateFrame",
        [](int tileset,
           const pxr::GfMatrix4d& viewMatrix,
           const pxr::GfMatrix4d& projMatrix,
           double height,
           double width) { updateFrame(tileset, viewMatrix.data(), projMatrix.data(), height, width); });
    m.def("setGeoreferenceOrigin", &setGeoreferenceOrigin);
}

} // namespace Cesium

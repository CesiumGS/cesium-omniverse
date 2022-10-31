#include "OmniTileset.h"
#include "pyboost11.h"

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

void startup() {
    Cesium::OmniTileset::init();

    // Brisbane coordinate
    Cesium::Georeference::instance().setOrigin(CesiumGeospatial::Ellipsoid::WGS84.cartographicToCartesian(
        CesiumGeospatial::Cartographic(glm::radians(153.0260), glm::radians(-27.4705), 500.0)));
}

void shutdown() {
    Cesium::OmniTileset::shutdown();
}

PYBIND11_MODULE(CesiumOmniversePythonBindings, m) {
    m.doc() = "pybind11 cesium plugin"; // optional module docstring

    m.def("startup", &startup, "startup Cesium");
    m.def("shutdown", &shutdown, "shutdown Cesium");

    pybind11::class_<Cesium::OmniTileset>(m, "OmniTileset")
        .def(pybind11::init<const pxr::UsdStageRefPtr&, const std::string&>())
        .def(pybind11::init<const pxr::UsdStageRefPtr&, int64_t, const std::string&>())
        .def("updateFrame", &Cesium::OmniTileset::updateFrame);
}

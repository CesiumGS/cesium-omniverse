#include "cesium/omniverse/CesiumOmniverse.h"

#include <carb/BindingsPythonUtils.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec4d.h>

// Needs to go after carb
#include "pyboost11.h"

namespace pybind11 {
namespace detail {

PYBOOST11_TYPE_CASTER(pxr::GfMatrix4d, _("Matrix4d"));

} // end namespace detail
} // end namespace pybind11

CARB_BINDINGS("cesium.omniverse.python")
DISABLE_PYBIND11_DYNAMIC_CAST(cesium::omniverse::ICesiumOmniverseInterface)

PYBIND11_MODULE(CesiumOmniversePythonBindings, m) {

    using namespace cesium::omniverse;

    m.doc() = "pybind11 cesium.omniverse bindings";

    carb::defineInterfaceClass<ICesiumOmniverseInterface>(
        m, "ICesiumOmniverseInterface", "acquire_cesium_omniverse_interface", "release_cesium_omniverse_interface")
        .def("initialize", &ICesiumOmniverseInterface::initialize)
        .def("finalize", &ICesiumOmniverseInterface::finalize)
        .def("addTilesetUrl", &ICesiumOmniverseInterface::addTilesetUrl)
        .def("addTilesetIon", &ICesiumOmniverseInterface::addTilesetIon)
        .def("removeTileset", &ICesiumOmniverseInterface::removeTileset)
        .def("addIonRasterOverlay", &ICesiumOmniverseInterface::addIonRasterOverlay)
        .def("updateFrame", &ICesiumOmniverseInterface::updateFrame)
        .def("setGeoreferenceOrigin", &ICesiumOmniverseInterface::setGeoreferenceOrigin);
}

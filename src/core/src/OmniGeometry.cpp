#include "cesium/omniverse/OmniGeometry.h"

namespace cesium::omniverse {

OmniGeometry::OmniGeometry(pxr::SdfPath path, const OmniGeometryDefinition& geometryDefinition, bool debugRandomColors)
    : _path(path)
    , _geometryDefinition(geometryDefinition)
    , _debugRandomColors(debugRandomColors) {}

void OmniGeometry::setActive(bool active) {
    if (!active) {
        reset();
    }
}

pxr::SdfPath OmniGeometry::getPath() const {
    return _path;
}

const OmniGeometryDefinition& OmniGeometry::getGeometryDefinition() const {
    return _geometryDefinition;
}

}; // namespace cesium::omniverse

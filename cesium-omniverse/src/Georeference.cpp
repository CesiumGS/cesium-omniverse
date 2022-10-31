#include "Georeference.h"

#include <CesiumGeometry/AxisTransforms.h>
#include <CesiumGeospatial/Transforms.h>

namespace Cesium {
Georeference& Georeference::instance() {
    static Georeference georef;
    return georef;
}

void Georeference::setOrigin(const glm::dvec3& origin) {
    relToAbsWorld =
        CesiumGeospatial::Transforms::eastNorthUpToFixedFrame(origin) * CesiumGeometry::AxisTransforms::Y_UP_TO_Z_UP;
    absToRelWorld = glm::inverse(relToAbsWorld);
    originChangeEvent.invoke(relToAbsWorld, absToRelWorld);
}
} // namespace Cesium

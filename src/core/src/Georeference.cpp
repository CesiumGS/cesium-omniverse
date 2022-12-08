#include "cesium/omniverse/Georeference.h"

#include <CesiumGeometry/AxisTransforms.h>
#include <CesiumGeospatial/Transforms.h>
#include <glm/gtc/matrix_transform.hpp>

namespace cesium::omniverse {
Georeference& Georeference::instance() {
    static Georeference georef;
    return georef;
}

void Georeference::setOrigin(const glm::dvec3& origin) {
    const auto centimeterToMeter = glm::scale(glm::dmat4(1.0), glm::dvec3(0.01));
    relToAbsWorld = CesiumGeospatial::Transforms::eastNorthUpToFixedFrame(origin) *
                    CesiumGeometry::AxisTransforms::Y_UP_TO_Z_UP * centimeterToMeter;
    absToRelWorld = glm::inverse(relToAbsWorld);
    originChangeEvent.invoke(relToAbsWorld, absToRelWorld);
}
} // namespace cesium::omniverse

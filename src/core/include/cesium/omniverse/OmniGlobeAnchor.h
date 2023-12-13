#pragma once

#include "cesium/omniverse/OmniGeoreference.h"

#include <CesiumGeospatial/Cartographic.h>
#include <CesiumGeospatial/GlobeAnchor.h>
#include <glm/ext/matrix_double4x4.hpp>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/usd/sdf/path.h>

namespace cesium::omniverse {

struct OmniGlobeAnchorValueCache {};

class OmniGlobeAnchor {
  public:
    OmniGlobeAnchor(const pxr::SdfPath& primPath, const CesiumGeospatial::Ellipsoid& ellipsoid);

    void updateByGeographicCoordinates();
    void updateByLocalTransform();
    void updateByFixedTransform();

    void updateOrigin();

    // TODO: move
    [[nodiscard]] bool getDetectTransformChanges() const;
    [[nodiscard]] pxr::SdfPath getGeoreferencePath() const;

  private:
    // Query from USD
    [[nodiscard]] bool getAdjustOrientation() const;
    [[nodiscard]] CesiumGeospatial::Cartographic getGeographicCoordinates() const;
    [[nodiscard]] glm::dvec3 getEcefPosition() const;
    [[nodiscard]] glm::dvec3 getEcefRotation() const;
    [[nodiscard]] glm::dvec3 getEcefScale() const;

    // Update USD
    void updateGeographicCoordinates();
    void updateEcefPositionRotationScale();
    void updateTransformOp();

    void updateCachedValues();

    void updateByGeographicCoordinatesInternal();
    void updateByLocalTransformInternal();
    void updateByFixedTransformInternal(
        const glm::dvec3& ecefPosition,
        const glm::dvec3& ecefRotation,
        const glm::dvec3& ecefScale);

    [[nodiscard]] glm::dmat4 getAnchorToLocalTransform() const;
    [[nodiscard]] CesiumGeospatial::Cartographic getOrigin() const;

    pxr::SdfPath _primPath;
    const CesiumGeospatial::Ellipsoid& _ellipsoid;
    std::shared_ptr<CesiumGeospatial::GlobeAnchor> _anchor;

    // These are used for quick comparisons, so we can short circuit successive updates.
    glm::dmat4 _cachedLocalTransform{};
    CesiumGeospatial::Cartographic _cachedGeographicCoordinates{0.0, 0.0, 0.0};
    glm::dvec3 _cachedEcefPosition{};
    glm::dvec3 _cachedEcefRotation{};
    glm::dvec3 _cachedEcefScale{};
};

} // namespace cesium::omniverse

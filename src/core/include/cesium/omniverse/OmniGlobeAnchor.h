#pragma once

#include <CesiumGeospatial/Cartographic.h>
#include <CesiumGeospatial/GlobeAnchor.h>
#include <glm/ext/matrix_double4x4.hpp>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/usd/sdf/path.h>

namespace cesium::omniverse {

struct OmniGlobeAnchorValueCache {
    pxr::GfMatrix4d transformation;
    pxr::GfVec3d geographicCoordinate;
    pxr::GfVec3d ecefPosition;
    pxr::GfVec3d ecefRotation;
    pxr::GfVec3d ecefScale;
};

class OmniGlobeAnchor {
  public:
    OmniGlobeAnchor(pxr::SdfPath anchorPrimPath, const glm::dmat4 anchorToFixed);

    [[nodiscard]] const pxr::GfMatrix4d& getCachedTransformation() const;
    [[nodiscard]] const pxr::GfVec3d& getCachedGeographicCoordinate() const;
    [[nodiscard]] const pxr::GfVec3d& getCachedEcefPosition() const;
    [[nodiscard]] const pxr::GfVec3d& getCachedEcefRotation() const;
    [[nodiscard]] const pxr::GfVec3d& getCachedEcefScale() const;
    void updateCachedValues();

    // Using cesium-native nomenclature here. Fixed corresponds to ECEF in our code.
    [[nodiscard]] const glm::dmat4& getAnchorToFixedTransform() const;

    // Using cesium-native nomenclature here. Local refers to ENU in our code.
    const glm::dmat4 getAnchorToLocalTransform(const CesiumGeospatial::Cartographic& origin);
    std::optional<CesiumGeospatial::Cartographic> getCartographicPosition();
    [[nodiscard]] const pxr::SdfPath& getPrimPath() const;
    void updateByFixedTransform(
        const glm::dvec3& ecefPositionVec,
        const glm::dvec3& ecefRotationVec,
        const glm::dvec3& ecefScaleVec,
        bool shouldReorient);
    void updateByGeographicCoordinates(double latitude, double longitude, double height, bool shouldReorient);
    void updateByUsdTransform(const CesiumGeospatial::Cartographic& origin, bool shouldReorient);

  private:
    pxr::SdfPath _anchorPrimPath;
    std::shared_ptr<CesiumGeospatial::GlobeAnchor> _anchor;

    // This is used for quick comparisons, so we can short circuit successive updates.
    OmniGlobeAnchorValueCache _valueCache;
};

} // namespace cesium::omniverse

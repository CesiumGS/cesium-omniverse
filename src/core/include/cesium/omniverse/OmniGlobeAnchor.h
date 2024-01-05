#pragma once

#include <CesiumGeospatial/Cartographic.h>
#include <glm/glm.hpp>
#include <pxr/usd/sdf/path.h>

namespace CesiumGeospatial {
class GlobeAnchor;
class Ellipsoid;
} // namespace CesiumGeospatial

namespace cesium::omniverse {

class Context;

class OmniGlobeAnchor {
  public:
    OmniGlobeAnchor(Context* pContext, const PXR_NS::SdfPath& path);
    ~OmniGlobeAnchor();
    OmniGlobeAnchor(const OmniGlobeAnchor&) = delete;
    OmniGlobeAnchor& operator=(const OmniGlobeAnchor&) = delete;
    OmniGlobeAnchor(OmniGlobeAnchor&&) noexcept = default;
    OmniGlobeAnchor& operator=(OmniGlobeAnchor&&) noexcept = default;

    [[nodiscard]] const PXR_NS::SdfPath& getPath() const;
    [[nodiscard]] bool getDetectTransformChanges() const;
    [[nodiscard]] bool getAdjustOrientation() const;
    [[nodiscard]] PXR_NS::SdfPath getGeoreferencePath() const;

    void updateByGeographicCoordinates();
    void updateByPrimLocalTransform();
    void updateByPrimLocalToEcefTransform();
    void updateByGeoreference();

  private:
    [[nodiscard]] bool isGeoreferenceValid() const;
    [[nodiscard]] bool initialize();
    void finalize();

    [[nodiscard]] glm::dmat4 getPrimLocalTransform() const;
    [[nodiscard]] CesiumGeospatial::Cartographic getGeographicCoordinates() const;
    [[nodiscard]] glm::dvec3 getPrimLocalToEcefTranslation() const;
    [[nodiscard]] glm::dvec3 getPrimLocalToEcefRotation() const;
    [[nodiscard]] glm::dvec3 getPrimLocalToEcefScale() const;

    void saveGeographicCoordinates();
    void savePrimLocalToEcefTransform();
    void savePrimLocalTransform();

    Context* _pContext;
    PXR_NS::SdfPath _path;
    std::unique_ptr<CesiumGeospatial::GlobeAnchor> _pAnchor;

    // These are used for quick comparisons, so we can short circuit successive updates.
    glm::dmat4 _cachedPrimLocalTransform{0.0};
    CesiumGeospatial::Cartographic _cachedGeographicCoordinates{0.0, 0.0, 0.0};
    glm::dvec3 _cachedPrimLocalToEcefTranslation{0.0};
    glm::dvec3 _cachedPrimLocalToEcefRotation{0.0};
    glm::dvec3 _cachedPrimLocalToEcefScale{1.0};
};

} // namespace cesium::omniverse

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
    OmniGlobeAnchor(Context* pContext, const pxr::SdfPath& path);
    ~OmniGlobeAnchor();
    OmniGlobeAnchor(const OmniGlobeAnchor&) = delete;
    OmniGlobeAnchor& operator=(const OmniGlobeAnchor&) = delete;
    OmniGlobeAnchor(OmniGlobeAnchor&&) noexcept = default;
    OmniGlobeAnchor& operator=(OmniGlobeAnchor&&) noexcept = default;

    [[nodiscard]] const pxr::SdfPath& getPath() const;
    [[nodiscard]] bool getDetectTransformChanges() const;
    [[nodiscard]] bool getAdjustOrientation() const;
    [[nodiscard]] pxr::SdfPath getResolvedGeoreferencePath() const;

    void updateByPrimLocalToEcefTransform();
    void updateByGeographicCoordinates();
    void updateByPrimLocalTransform();
    void updateByGeoreference();

  private:
    [[nodiscard]] bool isAnchorValid() const;
    void initialize();
    void finalize();

    [[nodiscard]] glm::dvec3 getPrimLocalToEcefTranslation() const;
    [[nodiscard]] glm::dvec3 getPrimLocalToEcefRotation() const;
    [[nodiscard]] glm::dvec3 getPrimLocalToEcefScale() const;
    [[nodiscard]] CesiumGeospatial::Cartographic getGeographicCoordinates() const;
    [[nodiscard]] glm::dvec3 getPrimLocalTranslation() const;
    [[nodiscard]] glm::dvec3 getPrimLocalRotation() const;
    [[nodiscard]] glm::dvec3 getPrimLocalScale() const;

    void savePrimLocalToEcefTransform();
    void saveGeographicCoordinates();
    void savePrimLocalTransform();

    Context* _pContext;
    pxr::SdfPath _path;
    std::unique_ptr<CesiumGeospatial::GlobeAnchor> _pAnchor;

    // These are used for quick comparisons, so we can short circuit successive updates.
    glm::dvec3 _cachedPrimLocalToEcefTranslation{0.0};
    glm::dvec3 _cachedPrimLocalToEcefRotation{0.0};
    glm::dvec3 _cachedPrimLocalToEcefScale{1.0};
    CesiumGeospatial::Cartographic _cachedGeographicCoordinates{0.0, 0.0, 0.0};
    glm::dvec3 _cachedPrimLocalTranslation{0.0};
    glm::dvec3 _cachedPrimLocalRotation{0.0, 0.0, 0.0};
    glm::dvec3 _cachedPrimLocalScale{1.0};
};

} // namespace cesium::omniverse

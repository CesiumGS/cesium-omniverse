#pragma once

#include <CesiumGeospatial/Ellipsoid.h>
#include <glm/ext/matrix_double4x4.hpp>
#include <pxr/usd/sdf/path.h>

#include <memory>
#include <optional>
#include <unordered_map>

namespace CesiumGeospatial {
class GlobeAnchor;
}

namespace cesium::omniverse {

class OmniGlobeAnchor;

class GlobeAnchorRegistry {
  public:
    GlobeAnchorRegistry(const GlobeAnchorRegistry&) = delete;
    GlobeAnchorRegistry(GlobeAnchorRegistry&&) = delete;

    static GlobeAnchorRegistry& getInstance() {
        static GlobeAnchorRegistry instance;
        return instance;
    }

    GlobeAnchorRegistry& operator=(const GlobeAnchorRegistry&) = delete;
    GlobeAnchorRegistry& operator=(GlobeAnchorRegistry) = delete;

    void clear();
    bool anchorExists(pxr::SdfPath path) const;
    std::shared_ptr<OmniGlobeAnchor> createAnchor(const pxr::SdfPath& path, const CesiumGeospatial::Ellipsoid& ellipsoid);
    std::optional<std::shared_ptr<OmniGlobeAnchor>> getAnchor(const pxr::SdfPath& path) const;
    std::vector<std::shared_ptr<OmniGlobeAnchor>> getAllAnchors() const;
    std::vector<std::string> getAllAnchorPaths() const;
    bool removeAnchor(const pxr::SdfPath& path);

  protected:
    GlobeAnchorRegistry() = default;
    ~GlobeAnchorRegistry() = default;

  private:
    std::unordered_map<std::string, std::shared_ptr<OmniGlobeAnchor>> _anchors{};
};
} // namespace cesium::omniverse

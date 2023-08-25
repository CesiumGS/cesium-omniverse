#pragma once

#include "OmniGlobeAnchor.h"

#include <CesiumGeospatial/GlobeAnchor.h>
#include <pxr/usd/sdf/path.h>

#include <memory>
#include <unordered_map>

namespace CesiumGeospatial {
class GlobeAnchor;
}

namespace cesium::omniverse {

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
    std::shared_ptr<OmniGlobeAnchor> createAnchor(pxr::SdfPath path, glm::dmat4 anchorToFixed);
    std::optional<std::shared_ptr<OmniGlobeAnchor>> getAnchor(const pxr::SdfPath& path);
    std::vector<std::shared_ptr<OmniGlobeAnchor>> getAllAnchors();

  protected:
    GlobeAnchorRegistry() = default;
    ~GlobeAnchorRegistry() = default;

  private:
    std::unordered_map<std::string, std::shared_ptr<OmniGlobeAnchor>> _anchors{};
};
} // namespace cesium::omniverse

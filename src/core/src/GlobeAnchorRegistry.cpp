#include "cesium/omniverse/GlobeAnchorRegistry.h"

#include <CesiumGeospatial/GlobeAnchor.h>

#include <memory>
#include <optional>

namespace cesium::omniverse {

void GlobeAnchorRegistry::clear() {
    _anchors.clear();
}

std::shared_ptr<CesiumGeospatial::GlobeAnchor>
GlobeAnchorRegistry::createAnchor(pxr::SdfPath path, glm::dmat4 anchorToFixed) {
    auto anchor = std::make_shared<CesiumGeospatial::GlobeAnchor>(anchorToFixed);

    _anchors.emplace(path.GetString(), anchor);

    return anchor;
}

std::optional<std::shared_ptr<CesiumGeospatial::GlobeAnchor>> GlobeAnchorRegistry::getAnchor(const pxr::SdfPath& path) {
    if (auto anchor = _anchors.find(path.GetString()); anchor != _anchors.end()) {
        return anchor->second;
    }

    return std::nullopt;
}

} // namespace cesium::omniverse

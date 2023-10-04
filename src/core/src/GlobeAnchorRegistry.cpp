#include "cesium/omniverse/GlobeAnchorRegistry.h"

#include <memory>
#include <optional>

namespace cesium::omniverse {

bool GlobeAnchorRegistry::anchorExists(pxr::SdfPath path) {
    return _anchors.count(path.GetText()) > 0;
}

void GlobeAnchorRegistry::clear() {
    _anchors.clear();
}

std::shared_ptr<OmniGlobeAnchor> GlobeAnchorRegistry::createAnchor(pxr::SdfPath path, glm::dmat4 anchorToFixed) {
    auto anchor = std::make_shared<OmniGlobeAnchor>(path, anchorToFixed);

    _anchors.emplace(path.GetString(), anchor);

    return anchor;
}

std::optional<std::shared_ptr<OmniGlobeAnchor>> GlobeAnchorRegistry::getAnchor(const pxr::SdfPath& path) {
    if (auto anchor = _anchors.find(path.GetString()); anchor != _anchors.end()) {
        return anchor->second;
    }

    return std::nullopt;
}

std::vector<std::shared_ptr<OmniGlobeAnchor>> GlobeAnchorRegistry::getAllAnchors() {
    std::vector<std::shared_ptr<OmniGlobeAnchor>> result;
    result.reserve(_anchors.size());

    for (const auto& item : _anchors) {
        result.push_back(item.second);
    }

    return result;
}

bool GlobeAnchorRegistry::removeAnchor(const pxr::SdfPath& path) {
    return _anchors.erase(path.GetText()) > 0;
}

} // namespace cesium::omniverse

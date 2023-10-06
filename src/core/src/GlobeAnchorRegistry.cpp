#include "cesium/omniverse/GlobeAnchorRegistry.h"

#include "cesium/omniverse/OmniGlobeAnchor.h"

#include <memory>
#include <optional>

namespace cesium::omniverse {

bool GlobeAnchorRegistry::anchorExists(pxr::SdfPath path) const {
    return _anchors.count(path.GetText()) > 0;
}

void GlobeAnchorRegistry::clear() {
    _anchors.clear();
}

std::shared_ptr<OmniGlobeAnchor>
GlobeAnchorRegistry::createAnchor(const pxr::SdfPath& path, const glm::dmat4& anchorToFixed) {
    auto anchor = std::make_shared<OmniGlobeAnchor>(path, anchorToFixed);

    _anchors.emplace(path.GetString(), anchor);

    return anchor;
}

std::optional<std::shared_ptr<OmniGlobeAnchor>> GlobeAnchorRegistry::getAnchor(const pxr::SdfPath& path) const {
    if (auto anchor = _anchors.find(path.GetString()); anchor != _anchors.end()) {
        return anchor->second;
    }

    return std::nullopt;
}

std::vector<std::shared_ptr<OmniGlobeAnchor>> GlobeAnchorRegistry::getAllAnchors() const {
    std::vector<std::shared_ptr<OmniGlobeAnchor>> result;
    result.reserve(_anchors.size());

    for (const auto& item : _anchors) {
        result.emplace_back(item.second);
    }

    return result;
}

std::vector<std::string> GlobeAnchorRegistry::getAllAnchorPaths() const {
    std::vector<std::string> result;
    result.reserve(_anchors.size());

    for (const auto& item : _anchors) {
        result.emplace_back(item.first);
    }

    return result;
}

bool GlobeAnchorRegistry::removeAnchor(const pxr::SdfPath& path) {
    return _anchors.erase(path.GetText()) > 0;
}

} // namespace cesium::omniverse

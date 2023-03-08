#include "cesium/omniverse/AssetRegistry.h"

#include "cesium/omniverse/OmniImagery.h"
#include "cesium/omniverse/OmniTileset.h"

namespace cesium::omniverse {

void AssetRegistry::addTileset(const pxr::SdfPath& path) {
    _tilesets.insert(_tilesets.end(), std::make_shared<OmniTileset>(path));
}

void AssetRegistry::removeTileset(const pxr::SdfPath& path) {
    _tilesets.remove_if([path](const auto& tileset) { return tileset->getPath() == path; });
}

std::optional<std::shared_ptr<OmniTileset>> AssetRegistry::getTilesetByPath(const pxr::SdfPath& path) const {
    for (const auto& tileset : _tilesets) {
        if (tileset->getPath() == path) {
            return tileset;
        }
    }

    return std::nullopt;
}

std::optional<std::shared_ptr<OmniTileset>> AssetRegistry::getTilesetByIonAssetId(int64_t ionAssetId) const {
    for (const auto& tileset : _tilesets) {
        if (tileset->getIonAssetId() == ionAssetId) {
            return std::make_optional(tileset);
        }
    }

    return std::nullopt;
}

const std::list<std::shared_ptr<OmniTileset>>& AssetRegistry::getAllTilesets() const {
    return _tilesets;
}

std::vector<pxr::SdfPath> AssetRegistry::getAllTilesetPaths() const {
    std::vector<pxr::SdfPath> result;
    result.reserve(_tilesets.size());

    for (const auto& tileset : _tilesets) {
        result.emplace_back(tileset->getPath());
    }

    return result;
}

void AssetRegistry::addImagery(const pxr::SdfPath& path) {
    _imageries.insert(_imageries.end(), std::make_shared<OmniImagery>(path));
}

std::optional<std::shared_ptr<OmniImagery>> AssetRegistry::getImageryByPath(const pxr::SdfPath& path) const {
    for (const auto& imagery : _imageries) {
        if (imagery->getPath() == path) {
            return imagery;
        }
    }

    return std::nullopt;
}

std::optional<std::shared_ptr<OmniImagery>> AssetRegistry::getImageryByIonAssetId(int64_t ionAssetId) const {
    for (const auto& imagery : _imageries) {
        if (imagery->getIonAssetId() == ionAssetId) {
            return imagery;
        }
    }

    return std::nullopt;
}

AssetType AssetRegistry::getAssetType(const pxr::SdfPath& path) const {
    if (getTilesetByPath(path).has_value()) {
        return AssetType::TILESET;
    }

    if (getImageryByPath(path).has_value()) {
        return AssetType::IMAGERY;
    }

    return AssetType::OTHER;
}

void AssetRegistry::clear() {
    _tilesets.clear();
    _imageries.clear();
}

} // namespace cesium::omniverse

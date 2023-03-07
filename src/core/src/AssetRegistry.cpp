#include "cesium/omniverse/AssetRegistry.h"

#include "cesium/omniverse/OmniIonRasterOverlay.h"
#include "cesium/omniverse/OmniTileset.h"

namespace cesium::omniverse {

void AssetRegistry::addTileset(const pxr::SdfPath& path, int64_t id) {
    _tilesets.insert(_tilesets.end(), std::make_shared<OmniTileset>(path, id));
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

void AssetRegistry::addRasterOverlay(const pxr::SdfPath& path) {
    _rasterOverlays.insert(_rasterOverlays.end(), std::make_shared<OmniIonRasterOverlay>(path));
}

std::optional<std::shared_ptr<OmniIonRasterOverlay>>
AssetRegistry::getRasterOverlayByPath(const pxr::SdfPath& path) const {
    for (const auto& rasterOverlay : _rasterOverlays) {
        if (rasterOverlay->getPath() == path) {
            return std::make_optional(rasterOverlay);
        }
    }

    return std::nullopt;
}

std::optional<std::shared_ptr<OmniIonRasterOverlay>>
AssetRegistry::getRasterOverlayByIonAssetId(int64_t ionAssetId) const {
    for (const auto& rasterOverlay : _rasterOverlays) {
        if (rasterOverlay->getIonAssetId() == ionAssetId) {
            return std::make_optional(rasterOverlay);
        }
    }

    return std::nullopt;
}

AssetType AssetRegistry::getAssetType(const pxr::SdfPath& path) const {
    if (getTilesetByPath(path).has_value()) {
        return AssetType::TILESET;
    } else if (getRasterOverlayByPath(path).has_value()) {
        return AssetType::RASTER_OVERLAY;
    }

    return AssetType::OTHER;
}

void AssetRegistry::clear() {
    _tilesets.clear();
    _rasterOverlays.clear();
}

} // namespace cesium::omniverse

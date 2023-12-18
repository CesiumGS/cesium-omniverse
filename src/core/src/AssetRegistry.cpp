#include "cesium/omniverse/AssetRegistry.h"

#include "cesium/omniverse/OmniImagery.h"
#include "cesium/omniverse/OmniTileset.h"

namespace cesium::omniverse {

void AssetRegistry::addTileset(const pxr::SdfPath& tilesetPath, const pxr::SdfPath& georeferencePath) {
    _tilesets.insert(_tilesets.end(), std::make_shared<OmniTileset>(tilesetPath, georeferencePath));
}

void AssetRegistry::removeTileset(const pxr::SdfPath& path) {
    _tilesets.remove_if([&path](const auto& tileset) { return tileset->getPath() == path; });
}

std::shared_ptr<OmniTileset> AssetRegistry::getTileset(const pxr::SdfPath& path) const {
    for (const auto& tileset : _tilesets) {
        if (tileset->getPath() == path) {
            return tileset;
        }
    }

    return nullptr;
}

const std::list<std::shared_ptr<OmniTileset>>& AssetRegistry::getAllTilesets() const {
    return _tilesets;
}

void AssetRegistry::addImagery(const pxr::SdfPath& path) {
    _imageries.insert(_imageries.end(), std::make_shared<OmniImagery>(path));
}

void AssetRegistry::removeImagery(const pxr::SdfPath& path) {
    _imageries.remove_if([&path](const auto& imagery) { return imagery->getPath() == path; });
}

std::shared_ptr<OmniImagery> AssetRegistry::getImagery(const pxr::SdfPath& path) const {
    for (const auto& imagery : _imageries) {
        if (imagery->getPath() == path) {
            return imagery;
        }
    }

    return nullptr;
}

std::shared_ptr<OmniImagery> AssetRegistry::getImageryByIonAssetId(int64_t ionAssetId) const {
    for (const auto& imagery : _imageries) {
        if (imagery->getIonAssetId() == ionAssetId) {
            return imagery;
        }
    }

    return nullptr;
}

const std::list<std::shared_ptr<OmniImagery>>& AssetRegistry::getAllImageries() const {
    return _imageries;
}

AssetType AssetRegistry::getAssetType(const pxr::SdfPath& path) const {
    if (getTileset(path) != nullptr) {
        return AssetType::TILESET;
    }

    if (getImagery(path) != nullptr) {
        return AssetType::IMAGERY;
    }

    return AssetType::OTHER;
}

void AssetRegistry::clear() {
    _tilesets.clear();
    _imageries.clear();
}

} // namespace cesium::omniverse

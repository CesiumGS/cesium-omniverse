#include "cesium/omniverse/AssetRegistry.h"

#include "cesium/omniverse/OmniIonImagery.h"
#include "cesium/omniverse/OmniTileset.h"

#include <memory>

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

void AssetRegistry::addIonImagery(const pxr::SdfPath& path) {
    _ionImageries.insert(_ionImageries.end(), std::make_shared<OmniIonImagery>(path));
}

void AssetRegistry::addPolygonImagery(const pxr::SdfPath& path) {
    _polygonImageries.insert(_polygonImageries.end(), std::make_shared<OmniPolygonImagery>(path));
}

void AssetRegistry::removeIonImagery(const pxr::SdfPath& path) {
    _ionImageries.remove_if([&path](const auto& imagery) { return imagery->getPath() == path; });
}

void AssetRegistry::removePolygonImagery(const pxr::SdfPath& path) {
    _polygonImageries.remove_if([&path](const auto& imagery) { return imagery->getPath() == path; });
}

std::shared_ptr<OmniIonImagery> AssetRegistry::getIonImagery(const pxr::SdfPath& path) const {
    for (const auto& imagery : _ionImageries) {
        if (imagery->getPath() == path) {
            return imagery;
        }
    }

    return nullptr;
}

std::shared_ptr<OmniPolygonImagery> AssetRegistry::getPolygonImagery(const pxr::SdfPath& path) const {
    for (const auto& imagery : _polygonImageries) {
        if (imagery->getPath() == path) {
            return imagery;
        }
    }

    return nullptr;
}

std::shared_ptr<OmniIonImagery> AssetRegistry::getImageryByIonAssetId(int64_t ionAssetId) const {
    for (const auto& ionImagery : _ionImageries) {
        if (ionImagery->getIonAssetId() == ionAssetId) {
            return ionImagery;
        }
    }

    return nullptr;
}

const std::list<std::shared_ptr<OmniIonImagery>> AssetRegistry::getAllIonImageries() const {
    return _ionImageries;
}

const std::list<std::shared_ptr<OmniPolygonImagery>> AssetRegistry::getAllPolygonImageries() const {
    return _polygonImageries;
}

AssetType AssetRegistry::getAssetType(const pxr::SdfPath& path) const {
    if (getTileset(path) != nullptr) {
        return AssetType::TILESET;
    }

    if (getIonImagery(path) != nullptr) {
        return AssetType::ION_IMAGERY;
    }

    if (getPolygonImagery(path) != nullptr) {
        return AssetType::POLYGON_IMAGERY;
    }

    return AssetType::OTHER;
}

void AssetRegistry::clear() {
    _tilesets.clear();
    _ionImageries.clear();
    _polygonImageries.clear();
}

} // namespace cesium::omniverse

#include "cesium/omniverse/AssetRegistry.h"

#include "cesium/omniverse/OmniIonRasterOverlay.h"
#include "cesium/omniverse/OmniTileset.h"

namespace cesium::omniverse {

void AssetRegistry::addTileset(int64_t id, const pxr::SdfPath& path) {
    std::string sPath{path.GetString().c_str()};
    auto item = AssetRegistryItem{id, AssetType::TILESET, std::make_shared<OmniTileset>(id, path), sPath, std::nullopt};
    items.insert(items.end(), item);
}

std::optional<std::shared_ptr<OmniTileset>> AssetRegistry::getTileset(int64_t assetId) {
    for (const auto& item : items) {
        if (item.assetId == assetId && item.type == AssetType::TILESET) {
            if (!item.tileset.has_value()) {
                return std::nullopt;
            }

            return item.tileset.value();
        }
    }

    return std::nullopt;
}

std::optional<std::shared_ptr<OmniTileset>> AssetRegistry::getTileset(const std::string& path) {
    for (const auto& item : items) {
        if (item.path == path && item.type == AssetType::TILESET) {
            if (!item.tileset.has_value()) {
                return std::nullopt;
            }

            return item.tileset.value();
        }
    }

    return std::nullopt;
}

std::optional<std::shared_ptr<OmniTileset>> AssetRegistry::getTilesetFromRasterOverlay(const std::string& path) {
    for (const auto& item : items) {
        if (item.path == path && item.type == AssetType::IMAGERY) {
            const auto& tilesetId = item.parentId.value();
            return getTileset(tilesetId);
        }
    }

    return std::nullopt;
}

std::optional<int64_t> AssetRegistry::getTilesetId(const std::string& path) {
    auto tileset = getTileset(path);

    if (!tileset.has_value()) {
        return std::nullopt;
    }

    return tileset.value()->getId();
}

std::vector<std::shared_ptr<OmniTileset>> AssetRegistry::getAllTilesets() {
    auto tilesets = std::vector<std::shared_ptr<OmniTileset>>();
    tilesets.reserve(size());

    for (const auto& item : items) {
        if (item.type == AssetType::TILESET && item.tileset.has_value()) {
            tilesets.emplace_back(item.tileset.value());
        }
    }

    return tilesets;
}

/**
 * @brief Gets all the tileset IDs and their paths. This is primarily for passing up to the python layer where we do not necessarily want to expose the tileset.
 *
 * @return A vector containing a pair with the assetId and path in that order.
 */
std::vector<std::pair<int64_t, const char*>> AssetRegistry::getAllTilesetIdsAndPaths() {
    std::vector<std::pair<int64_t, const char*>> result;
    result.reserve(size());

    for (const auto& item : items) {
        result.emplace_back(item.assetId, item.path.c_str());
    }

    return result;
}

[[maybe_unused]] std::vector<int64_t> AssetRegistry::getAllTilesetIds() {
    return getAssetIdsByType(AssetType::TILESET);
}

std::optional<const AssetRegistryItem> AssetRegistry::getItemByPath(const pxr::SdfPath& path) {
    const auto& strPath = path.GetString();
    for (const auto& item : items) {
        if (item.path == strPath) {
            return item;
        }
    }

    return std::nullopt;
}

void AssetRegistry::addRasterOverlay(int64_t assetId, const pxr::SdfPath& path, int64_t parentId) {
    items.insert(items.end(), AssetRegistryItem{assetId, AssetType::IMAGERY, std::nullopt, path.GetString(), parentId});
}

void AssetRegistry::setRasterOverlayAssetId(const pxr::SdfPath& path, int64_t assetId) {
    for (auto& item : items) {
        if (item.path == path.GetString()) {
            item.assetId = assetId;
            return;
        }
    }
}

std::optional<OmniIonRasterOverlay> AssetRegistry::getRasterOverlay(int64_t assetId) {
    for (const auto& item : items) {
        if (item.type == AssetType::IMAGERY && item.assetId == assetId) {
            return OmniIonRasterOverlay(pxr::SdfPath(item.path));
        }
    }

    return std::nullopt;
}

std::optional<int64_t> AssetRegistry::getRasterOverlayIdByPath(const pxr::SdfPath& path) {
    for (const auto& item : items) {
        if (item.type == AssetType::IMAGERY && item.path == path.GetString()) {
            return item.assetId;
        }
    }

    return std::nullopt;
}

[[maybe_unused]] std::vector<int64_t> AssetRegistry::getAllRasterOverlayIds() {
    return getAssetIdsByType(AssetType::IMAGERY);
}

[[maybe_unused]] std::vector<int64_t> AssetRegistry::getAllRasterOverlayIdsForTileset(int64_t parentId) {
    std::vector<int64_t> result;
    result.reserve(size());

    for (const auto& item : items) {
        if (item.type == AssetType::IMAGERY && item.parentId == parentId) {
            result.emplace_back(item.assetId);
        }
    }

    return result;
}

void AssetRegistry::clear() {
    items.clear();
}

void AssetRegistry::removeAsset(int64_t assetId) {
    items.remove_if([assetId](AssetRegistryItem& item) { return item.assetId == assetId; });
}

void AssetRegistry::removeAssetByParent(int64_t parentId) {
    items.remove_if([parentId](AssetRegistryItem& item) { return item.parentId == parentId; });
}

uint64_t AssetRegistry::size() {
    return items.size();
}

std::vector<int64_t> AssetRegistry::getAssetIdsByType(AssetType type) {
    auto result = std::vector<int64_t>();

    for (const auto& item : items) {
        if (item.type == type) {
            result.emplace_back(item.assetId);
        }
    }

    return result;
}

} // namespace cesium::omniverse

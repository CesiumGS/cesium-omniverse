#pragma once

#include "pxr/usd/sdf/path.h"

#include <list>

namespace cesium::omniverse {

class OmniTileset;

enum AssetType {
    TILESET = 0,
    IMAGERY,
};

struct AssetRegistryItem {
    int64_t assetId;
    AssetType type;
    std::optional<std::shared_ptr<OmniTileset>> tileset;
    std::string path;
    std::optional<int64_t> parentId;
};

class AssetRegistry {
  public:
    AssetRegistry(const AssetRegistry&) = delete;
    AssetRegistry(AssetRegistry&&) = delete;

    static AssetRegistry& getInstance() {
        static AssetRegistry instance;
        return instance;
    }

    AssetRegistry& operator=(const AssetRegistry&) = delete;
    AssetRegistry& operator=(AssetRegistry) = delete;

    void addTileset(int64_t id, const pxr::SdfPath& path);
    std::optional<std::shared_ptr<OmniTileset>> getTileset(int64_t assetId);
    std::optional<std::shared_ptr<OmniTileset>> getTileset(const std::string& path);
    std::optional<int64_t> getTilesetId(const std::string& path);
    std::vector<std::shared_ptr<OmniTileset>> getAllTilesets();
    std::optional<std::shared_ptr<OmniTileset>> getTilesetFromRasterOverlay(const std::string& path);
    std::vector<std::pair<int64_t, const char*>> getAllTilesetIdsAndPaths();
    [[maybe_unused]] std::vector<int64_t> getAllTilesetIds();

    void addRasterOverlay(int64_t assetId, const pxr::SdfPath& path, int64_t parentId);
    void setRasterOverlayAssetId(const pxr::SdfPath& path, int64_t assetId);
    [[maybe_unused]] std::vector<int64_t> getAllRasterOverlayIds();
    [[maybe_unused]] std::vector<int64_t> getAllRasterOverlayIdsForTileset(int64_t parentId);

    void clear();

    void removeAsset(int64_t assetId);
    void removeAssetByParent(int64_t parentId);

    uint64_t size();

  protected:
    AssetRegistry() = default;
    ~AssetRegistry() = default;

  private:
    std::list<AssetRegistryItem> items{};

    std::vector<int64_t> getAssetIdsByType(AssetType type);
};

} // namespace cesium::omniverse

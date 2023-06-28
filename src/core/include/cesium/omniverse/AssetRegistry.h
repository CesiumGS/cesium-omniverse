#pragma once

#include "pxr/usd/sdf/path.h"

#include <list>

namespace cesium::omniverse {

class OmniTileset;
class OmniImagery;

enum AssetType {
    TILESET = 0,
    IMAGERY,
    OTHER,
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

    void addTileset(const pxr::SdfPath& tilesetPath, const pxr::SdfPath& georeferencePath);
    void removeTileset(const pxr::SdfPath& path);
    [[nodiscard]] std::optional<std::shared_ptr<OmniTileset>> getTilesetByPath(const pxr::SdfPath& path) const;
    [[nodiscard]] std::optional<std::shared_ptr<OmniTileset>> getTilesetByIonAssetId(int64_t ionAssetId) const;
    [[nodiscard]] const std::list<std::shared_ptr<OmniTileset>>& getAllTilesets() const;
    [[nodiscard]] std::vector<pxr::SdfPath> getAllTilesetPaths() const;

    void addImagery(const pxr::SdfPath& path);
    void removeImagery(const pxr::SdfPath& path);
    [[nodiscard]] std::optional<std::shared_ptr<OmniImagery>> getImageryByPath(const pxr::SdfPath& path) const;
    [[nodiscard]] std::optional<std::shared_ptr<OmniImagery>> getImageryByIonAssetId(int64_t ionAssetId) const;
    [[nodiscard]] const std::list<std::shared_ptr<OmniImagery>>& getAllImageries() const;

    [[nodiscard]] AssetType getAssetType(const pxr::SdfPath& path) const;

    void clear();

  protected:
    AssetRegistry() = default;
    ~AssetRegistry() = default;

  private:
    std::list<std::shared_ptr<OmniTileset>> _tilesets{};
    std::list<std::shared_ptr<OmniImagery>> _imageries{};
};

} // namespace cesium::omniverse

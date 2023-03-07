#pragma once

#include "pxr/usd/sdf/path.h"

#include <list>

namespace cesium::omniverse {

class OmniTileset;
class OmniIonRasterOverlay;

enum AssetType {
    TILESET = 0,
    RASTER_OVERLAY,
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

    void addTileset(const pxr::SdfPath& path, int64_t id);
    void removeTileset(const pxr::SdfPath& path);
    std::optional<std::shared_ptr<OmniTileset>> getTilesetByPath(const pxr::SdfPath& path) const;
    std::optional<std::shared_ptr<OmniTileset>> getTilesetByIonAssetId(int64_t ionAssetId) const;
    const std::list<std::shared_ptr<OmniTileset>>& getAllTilesets() const;
    std::vector<pxr::SdfPath> getAllTilesetPaths() const;

    void addRasterOverlay(const pxr::SdfPath& path);
    std::optional<std::shared_ptr<OmniIonRasterOverlay>> getRasterOverlayByPath(const pxr::SdfPath& path) const;
    std::optional<std::shared_ptr<OmniIonRasterOverlay>> getRasterOverlayByIonAssetId(int64_t ionAssetId) const;

    AssetType getAssetType(const pxr::SdfPath& path) const;

    void clear();

  protected:
    AssetRegistry() = default;
    ~AssetRegistry() = default;

  private:
    std::list<std::shared_ptr<OmniTileset>> _tilesets{};
    std::list<std::shared_ptr<OmniIonRasterOverlay>> _rasterOverlays{};
};

} // namespace cesium::omniverse

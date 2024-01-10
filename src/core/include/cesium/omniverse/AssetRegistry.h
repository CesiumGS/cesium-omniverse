#pragma once

#include "cesium/omniverse/OmniPolygonImagery.h"
#include <pxr/usd/sdf/path.h>

#include <list>

namespace cesium::omniverse {

class OmniTileset;
class OmniImagery;
class OmniIonImagery;
class OmniPolygonImagery;

enum AssetType {
    TILESET,
    ION_IMAGERY,
    POLYGON_IMAGERY,
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
    [[nodiscard]] std::shared_ptr<OmniTileset> getTileset(const pxr::SdfPath& path) const;
    [[nodiscard]] const std::list<std::shared_ptr<OmniTileset>>& getAllTilesets() const;

    void addIonImagery(const pxr::SdfPath& path);
    void addPolygonImagery(const pxr::SdfPath& path);
    void removeIonImagery(const pxr::SdfPath& path);
    void removePolygonImagery(const pxr::SdfPath& path);
    [[nodiscard]] std::shared_ptr<OmniIonImagery> getIonImagery(const pxr::SdfPath& path) const;
    [[nodiscard]] std::shared_ptr<OmniPolygonImagery> getPolygonImagery(const pxr::SdfPath& path) const;
    [[nodiscard]] std::shared_ptr<OmniIonImagery> getImageryByIonAssetId(int64_t ionAssetId) const;
    [[nodiscard]] const std::list<std::shared_ptr<OmniIonImagery>> getAllIonImageries() const;
    [[nodiscard]] const std::list<std::shared_ptr<OmniPolygonImagery>> getAllPolygonImageries() const;

    [[nodiscard]] AssetType getAssetType(const pxr::SdfPath& path) const;

    void clear();

  protected:
    AssetRegistry() = default;
    ~AssetRegistry() = default;

  private:
    std::list<std::shared_ptr<OmniTileset>> _tilesets;
    std::list<std::shared_ptr<OmniIonImagery>> _ionImageries;
    std::list<std::shared_ptr<OmniPolygonImagery>> _polygonImageries;
};

} // namespace cesium::omniverse

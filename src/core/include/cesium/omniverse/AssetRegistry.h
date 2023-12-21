#pragma once

#include <pxr/usd/sdf/path.h>

#include <list>

namespace cesium::omniverse {

class OmniTileset;
class OmniImagery;
class OmniIonImagery;

enum AssetType {
    TILESET,
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
    [[nodiscard]] std::shared_ptr<OmniTileset> getTileset(const pxr::SdfPath& path) const;
    [[nodiscard]] const std::list<std::shared_ptr<OmniTileset>>& getAllTilesets() const;

    void addImagery(const pxr::SdfPath& path);
    void removeImagery(const pxr::SdfPath& path);
    [[nodiscard]] std::shared_ptr<OmniImagery> getImagery(const pxr::SdfPath& path) const;
    [[nodiscard]] std::shared_ptr<OmniIonImagery> getImageryByIonAssetId(int64_t ionAssetId) const;
    [[nodiscard]] const std::list<std::shared_ptr<OmniImagery>>& getAllImageries() const;
    [[nodiscard]] const std::list<std::shared_ptr<OmniIonImagery>> getAllIonImageries() const;

    [[nodiscard]] AssetType getAssetType(const pxr::SdfPath& path) const;

    void clear();

  protected:
    AssetRegistry() = default;
    ~AssetRegistry() = default;

  private:
    std::list<std::shared_ptr<OmniTileset>> _tilesets;
    std::list<std::shared_ptr<OmniImagery>> _imageries;
};

} // namespace cesium::omniverse

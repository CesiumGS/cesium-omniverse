#pragma once

#include <pxr/usd/usd/common.h>

#include <vector>

#include <gsl/span>

namespace cesium::omniverse {

class Context;
class OmniCartographicPolygon;
class OmniData;
class OmniGeoreference;
class OmniGlobeAnchor;
class OmniImagery;
class OmniIonImagery;
class OmniIonServer;
class OmniPolygonImagery;
class OmniTileset;
struct Viewport;

enum AssetType {
    DATA,
    TILESET,
    ION_IMAGERY,
    POLYGON_IMAGERY,
    GEOREFERENCE,
    GLOBE_ANCHOR,
    ION_SERVER,
    CARTOGRAPHIC_POLYGON,
    OTHER,
};

class AssetRegistry {
  public:
    AssetRegistry(Context* pContext);
    ~AssetRegistry();
    AssetRegistry(const AssetRegistry&) = delete;
    AssetRegistry& operator=(const AssetRegistry&) = delete;
    AssetRegistry(AssetRegistry&&) noexcept = delete;
    AssetRegistry& operator=(AssetRegistry&&) noexcept = delete;

    void onUpdateFrame(const gsl::span<const Viewport>& viewports);

    OmniData& addData(const pxr::SdfPath& path);
    void removeData(const pxr::SdfPath& path);
    [[nodiscard]] OmniData* getData(const pxr::SdfPath& path) const;
    [[nodiscard]] OmniData* getFirstData() const;

    OmniTileset& addTileset(const pxr::SdfPath& path);
    void removeTileset(const pxr::SdfPath& path);
    [[nodiscard]] OmniTileset* getTileset(const pxr::SdfPath& path) const;
    [[nodiscard]] const std::vector<std::unique_ptr<OmniTileset>>& getTilesets() const;

    OmniIonImagery& addIonImagery(const pxr::SdfPath& path);
    void removeIonImagery(const pxr::SdfPath& path);
    [[nodiscard]] OmniIonImagery* getIonImagery(const pxr::SdfPath& path) const;
    [[nodiscard]] OmniIonImagery* getIonImageryByIonAssetId(int64_t ionAssetId) const;
    [[nodiscard]] const std::vector<std::unique_ptr<OmniIonImagery>>& getIonImageries() const;

    OmniPolygonImagery& addPolygonImagery(const pxr::SdfPath& path);
    void removePolygonImagery(const pxr::SdfPath& path);
    [[nodiscard]] OmniPolygonImagery* getPolygonImagery(const pxr::SdfPath& path) const;
    [[nodiscard]] const std::vector<std::unique_ptr<OmniPolygonImagery>>& getPolygonImageries() const;

    [[nodiscard]] OmniImagery* getImagery(const pxr::SdfPath& path) const;

    OmniGeoreference& addGeoreference(const pxr::SdfPath& path);
    void removeGeoreference(const pxr::SdfPath& path);
    [[nodiscard]] OmniGeoreference* getGeoreference(const pxr::SdfPath& path) const;
    [[nodiscard]] const std::vector<std::unique_ptr<OmniGeoreference>>& getGeoreferences() const;
    [[nodiscard]] OmniGeoreference* getFirstGeoreference() const;

    OmniGlobeAnchor& addGlobeAnchor(const pxr::SdfPath& path);
    void removeGlobeAnchor(const pxr::SdfPath& path);
    [[nodiscard]] OmniGlobeAnchor* getGlobeAnchor(const pxr::SdfPath& path) const;
    [[nodiscard]] const std::vector<std::unique_ptr<OmniGlobeAnchor>>& getGlobeAnchors() const;

    OmniIonServer& addIonServer(const pxr::SdfPath& path);
    void removeIonServer(const pxr::SdfPath& path);
    [[nodiscard]] OmniIonServer* getIonServer(const pxr::SdfPath& path) const;
    [[nodiscard]] const std::vector<std::unique_ptr<OmniIonServer>>& getIonServers() const;
    [[nodiscard]] OmniIonServer* getFirstIonServer() const;

    OmniCartographicPolygon& addCartographicPolygon(const pxr::SdfPath& path);
    void removeCartographicPolygon(const pxr::SdfPath& path);
    [[nodiscard]] OmniCartographicPolygon* getCartographicPolygon(const pxr::SdfPath& path) const;
    [[nodiscard]] const std::vector<std::unique_ptr<OmniCartographicPolygon>>& getCartographicPolygons() const;

    [[nodiscard]] AssetType getAssetType(const pxr::SdfPath& path) const;
    [[nodiscard]] bool hasAsset(const pxr::SdfPath& path) const;

    void clear();

  private:
    Context* _pContext;
    std::vector<std::unique_ptr<OmniData>> _datas;
    std::vector<std::unique_ptr<OmniTileset>> _tilesets;
    std::vector<std::unique_ptr<OmniIonImagery>> _ionImageries;
    std::vector<std::unique_ptr<OmniPolygonImagery>> _polygonImageries;
    std::vector<std::unique_ptr<OmniGeoreference>> _georeferences;
    std::vector<std::unique_ptr<OmniGlobeAnchor>> _globeAnchors;
    std::vector<std::unique_ptr<OmniIonServer>> _ionServers;
    std::vector<std::unique_ptr<OmniCartographicPolygon>> _cartographicPolygons;

    int64_t _tilesetId{0};
};

} // namespace cesium::omniverse

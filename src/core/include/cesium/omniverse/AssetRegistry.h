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

    OmniData& addData(const PXR_NS::SdfPath& path);
    void removeData(const PXR_NS::SdfPath& path);
    [[nodiscard]] OmniData* getData(const PXR_NS::SdfPath& path) const;
    [[nodiscard]] OmniData* getFirstData() const;

    OmniTileset& addTileset(const PXR_NS::SdfPath& path);
    void removeTileset(const PXR_NS::SdfPath& path);
    [[nodiscard]] OmniTileset* getTileset(const PXR_NS::SdfPath& path) const;
    [[nodiscard]] const std::vector<std::unique_ptr<OmniTileset>>& getAllTilesets() const;

    OmniIonImagery& addIonImagery(const PXR_NS::SdfPath& path);
    void removeIonImagery(const PXR_NS::SdfPath& path);
    [[nodiscard]] OmniIonImagery* getIonImagery(const PXR_NS::SdfPath& path) const;
    [[nodiscard]] OmniIonImagery* getIonImageryByIonAssetId(int64_t ionAssetId) const;
    [[nodiscard]] const std::vector<std::unique_ptr<OmniIonImagery>>& getAllIonImageries() const;

    OmniPolygonImagery& addPolygonImagery(const PXR_NS::SdfPath& path);
    void removePolygonImagery(const PXR_NS::SdfPath& path);
    [[nodiscard]] OmniPolygonImagery* getPolygonImagery(const PXR_NS::SdfPath& path) const;
    [[nodiscard]] const std::vector<std::unique_ptr<OmniPolygonImagery>>& getAllPolygonImageries() const;

    [[nodiscard]] OmniImagery* getImagery(const PXR_NS::SdfPath& path) const;

    OmniGeoreference& addGeoreference(const PXR_NS::SdfPath& path);
    void removeGeoreference(const PXR_NS::SdfPath& path);
    [[nodiscard]] OmniGeoreference* getGeoreference(const PXR_NS::SdfPath& path) const;
    [[nodiscard]] const std::vector<std::unique_ptr<OmniGeoreference>>& getAllGeoreferences() const;

    OmniGlobeAnchor& addGlobeAnchor(const PXR_NS::SdfPath& path);
    void removeGlobeAnchor(const PXR_NS::SdfPath& path);
    [[nodiscard]] OmniGlobeAnchor* getGlobeAnchor(const PXR_NS::SdfPath& path) const;
    [[nodiscard]] const std::vector<std::unique_ptr<OmniGlobeAnchor>>& getAllGlobeAnchors() const;

    OmniIonServer& addIonServer(const PXR_NS::SdfPath& path);
    void removeIonServer(const PXR_NS::SdfPath& path);
    [[nodiscard]] OmniIonServer* getIonServer(const PXR_NS::SdfPath& path) const;
    [[nodiscard]] const std::vector<std::unique_ptr<OmniIonServer>>& getAllIonServers() const;
    [[nodiscard]] OmniIonServer* getFirstIonServer() const;

    OmniCartographicPolygon& addCartographicPolygon(const PXR_NS::SdfPath& path);
    void removeCartographicPolygon(const PXR_NS::SdfPath& path);
    [[nodiscard]] OmniCartographicPolygon* getCartographicPolygon(const PXR_NS::SdfPath& path) const;
    [[nodiscard]] const std::vector<std::unique_ptr<OmniCartographicPolygon>>& getAllCartographicPolygons() const;

    [[nodiscard]] AssetType getAssetType(const PXR_NS::SdfPath& path) const;
    [[nodiscard]] bool hasAsset(const PXR_NS::SdfPath& path) const;

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

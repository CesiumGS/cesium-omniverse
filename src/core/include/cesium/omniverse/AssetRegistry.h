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
class OmniRasterOverlay;
class OmniIonRasterOverlay;
class OmniIonServer;
class OmniPolygonRasterOverlay;
class OmniTileset;
class OmniWebMapServiceRasterOverlay;
class OmniTileMapServiceRasterOverlay;
class OmniWebMapTileServiceRasterOverlay;
struct Viewport;

enum AssetType {
    DATA,
    TILESET,
    ION_RASTER_OVERLAY,
    POLYGON_RASTER_OVERLAY,
    WEB_MAP_SERVICE_RASTER_OVERLAY,
    TILE_MAP_SERVICE_RASTER_OVERLAY,
    WEB_MAP_TILE_SERVICE_RASTER_OVERLAY,
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

    void onUpdateFrame(const gsl::span<const Viewport>& viewports, bool waitForLoadingTiles);

    OmniData& addData(const pxr::SdfPath& path);
    void removeData(const pxr::SdfPath& path);
    [[nodiscard]] OmniData* getData(const pxr::SdfPath& path) const;
    [[nodiscard]] OmniData* getFirstData() const;

    OmniTileset& addTileset(const pxr::SdfPath& path);
    void removeTileset(const pxr::SdfPath& path);
    [[nodiscard]] OmniTileset* getTileset(const pxr::SdfPath& path) const;
    [[nodiscard]] const std::vector<std::unique_ptr<OmniTileset>>& getTilesets() const;

    OmniIonRasterOverlay& addIonRasterOverlay(const pxr::SdfPath& path);
    void removeIonRasterOverlay(const pxr::SdfPath& path);
    [[nodiscard]] OmniIonRasterOverlay* getIonRasterOverlay(const pxr::SdfPath& path) const;
    [[nodiscard]] OmniIonRasterOverlay* getIonRasterOverlayByIonAssetId(int64_t ionAssetId) const;
    [[nodiscard]] const std::vector<std::unique_ptr<OmniIonRasterOverlay>>& getIonRasterOverlays() const;

    OmniPolygonRasterOverlay& addPolygonRasterOverlay(const pxr::SdfPath& path);
    void removePolygonRasterOverlay(const pxr::SdfPath& path);
    [[nodiscard]] OmniPolygonRasterOverlay* getPolygonRasterOverlay(const pxr::SdfPath& path) const;
    [[nodiscard]] const std::vector<std::unique_ptr<OmniPolygonRasterOverlay>>& getPolygonRasterOverlays() const;

    OmniWebMapServiceRasterOverlay& addWebMapServiceRasterOverlay(const pxr::SdfPath& path);
    void removeWebMapServiceRasterOverlay(const pxr::SdfPath& path);
    [[nodiscard]] OmniWebMapServiceRasterOverlay* getWebMapServiceRasterOverlay(const pxr::SdfPath& path) const;
    // [[nodiscard]] const std::vector<std::unique_ptr<OmniWebMapServiceRasterOverlay>>&
    // getWebMapServiceRasterOverlays() const;

    OmniTileMapServiceRasterOverlay& addTileMapServiceRasterOverlay(const pxr::SdfPath& path);
    void removeTileMapServiceRasterOverlay(const pxr::SdfPath& path);
    [[nodiscard]] OmniTileMapServiceRasterOverlay* getTileMapServiceRasterOverlay(const pxr::SdfPath& path) const;
    [[nodiscard]] const std::vector<std::unique_ptr<OmniTileMapServiceRasterOverlay>>& getTileMapServiceRasterOverlays() const;

    OmniWebMapTileServiceRasterOverlay& addWebMapTileServiceRasterOverlay(const pxr::SdfPath& path);
    void removeWebMapTileServiceRasterOverlay(const pxr::SdfPath& path);
    [[nodiscard]] OmniWebMapTileServiceRasterOverlay* getWebMapTileServiceRasterOverlay(const pxr::SdfPath& path) const;

    [[nodiscard]] OmniRasterOverlay* getRasterOverlay(const pxr::SdfPath& path) const;

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
    std::vector<std::unique_ptr<OmniIonRasterOverlay>> _ionRasterOverlays;
    std::vector<std::unique_ptr<OmniPolygonRasterOverlay>> _polygonRasterOverlays;
    std::vector<std::unique_ptr<OmniWebMapServiceRasterOverlay>> _webMapServiceRasterOverlays;
    std::vector<std::unique_ptr<OmniTileMapServiceRasterOverlay>> _tileMapServiceRasterOverlays;
    std::vector<std::unique_ptr<OmniWebMapTileServiceRasterOverlay>> _webMapTileServiceRasterOverlays;
    std::vector<std::unique_ptr<OmniGeoreference>> _georeferences;
    std::vector<std::unique_ptr<OmniGlobeAnchor>> _globeAnchors;
    std::vector<std::unique_ptr<OmniIonServer>> _ionServers;
    std::vector<std::unique_ptr<OmniCartographicPolygon>> _cartographicPolygons;

    int64_t _tilesetId{0};
};

} // namespace cesium::omniverse

#pragma once

#include <glm/glm.hpp>
#include <pxr/usd/sdf/path.h>

#include <optional>

#include <gsl/span>

namespace Cesium3DTilesSelection {
class Tileset;
class ViewState;
class ViewUpdateResult;
} // namespace Cesium3DTilesSelection

namespace CesiumRasterOverlays {
class RasterOverlay;
}

namespace CesiumGeospatial {
class Ellipsoid;
}

namespace CesiumGltf {
struct Model;
}

namespace CesiumIonClient {
struct Token;
}

namespace cesium::omniverse {

class Context;
class FabricPrepareRenderResources;
class OmniRasterOverlay;
struct TilesetStatistics;
struct Viewport;

enum TilesetSourceType {
    ION,
    URL,
};

class OmniTileset {
  public:
    OmniTileset(Context* pContext, const pxr::SdfPath& path, int64_t tilesetId);
    ~OmniTileset();
    OmniTileset(const OmniTileset&) = delete;
    OmniTileset& operator=(const OmniTileset&) = delete;
    OmniTileset(OmniTileset&&) noexcept = default;
    OmniTileset& operator=(OmniTileset&&) noexcept = default;

    [[nodiscard]] const pxr::SdfPath& getPath() const;
    [[nodiscard]] int64_t getTilesetId() const;
    [[nodiscard]] TilesetStatistics getStatistics() const;
    [[nodiscard]] const CesiumGeospatial::Ellipsoid& getEllipsoid() const;

    [[nodiscard]] TilesetSourceType getSourceType() const;
    [[nodiscard]] std::string getUrl() const;
    [[nodiscard]] int64_t getIonAssetId() const;
    [[nodiscard]] CesiumIonClient::Token getIonAccessToken() const;
    [[nodiscard]] std::string getIonApiUrl() const;
    [[nodiscard]] pxr::SdfPath getResolvedIonServerPath() const;
    [[nodiscard]] double getMaximumScreenSpaceError() const;
    [[nodiscard]] bool getPreloadAncestors() const;
    [[nodiscard]] bool getPreloadSiblings() const;
    [[nodiscard]] bool getForbidHoles() const;
    [[nodiscard]] uint32_t getMaximumSimultaneousTileLoads() const;
    [[nodiscard]] uint64_t getMaximumCachedBytes() const;
    [[nodiscard]] uint32_t getLoadingDescendantLimit() const;
    [[nodiscard]] bool getEnableFrustumCulling() const;
    [[nodiscard]] bool getEnableFogCulling() const;
    [[nodiscard]] bool getEnforceCulledScreenSpaceError() const;
    [[nodiscard]] double getMainThreadLoadingTimeLimit() const;
    [[nodiscard]] double getCulledScreenSpaceError() const;
    [[nodiscard]] bool getSuspendUpdate() const;
    [[nodiscard]] bool getSmoothNormals() const;
    [[nodiscard]] bool getShowCreditsOnScreen() const;
    [[nodiscard]] pxr::SdfPath getResolvedGeoreferencePath() const;
    [[nodiscard]] pxr::SdfPath getMaterialPath() const;
    [[nodiscard]] glm::dvec3 getDisplayColor() const;
    [[nodiscard]] double getDisplayOpacity() const;
    [[nodiscard]] std::vector<pxr::SdfPath> getRasterOverlayPaths() const;
    [[nodiscard]] double getPointSize() const;

    void updateTilesetOptions();

    void reload();
    [[nodiscard]] pxr::SdfPath getRasterOverlayPathIfExists(const CesiumRasterOverlays::RasterOverlay& rasterOverlay);
    void updateRasterOverlayAlpha(const pxr::SdfPath& rasterOverlayPath);
    void updateShaderInput(const pxr::SdfPath& shaderPath, const pxr::TfToken& attributeName);
    void updateDisplayColorAndOpacity();
    void addRasterOverlayIfExists(const OmniRasterOverlay* overlay);

    void onUpdateFrame(const gsl::span<const Viewport>& viewports, bool waitForLoadingTiles);

  private:
    void updateTransform();
    void updateView(const gsl::span<const Viewport>& viewports, bool waitForLoadingTiles);
    [[nodiscard]] bool updateExtent();
    void updateLoadStatus();

    void destroyNativeTileset();

    std::unique_ptr<Cesium3DTilesSelection::Tileset> _pTileset;
    std::shared_ptr<FabricPrepareRenderResources> _pRenderResourcesPreparer;
    const Cesium3DTilesSelection::ViewUpdateResult* _pViewUpdateResult;

    Context* _pContext;
    pxr::SdfPath _path;
    int64_t _tilesetId;
    glm::dmat4 _ecefToPrimWorldTransform{};
    std::vector<Cesium3DTilesSelection::ViewState> _viewStates;
    bool _extentSet{false};
    bool _activeLoading{false};
};

} // namespace cesium::omniverse

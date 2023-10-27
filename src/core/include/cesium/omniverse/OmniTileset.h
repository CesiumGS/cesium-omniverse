#pragma once

#include <CesiumIonClient/Token.h>
#include <CesiumUsdSchemas/georeference.h>
#include <glm/glm.hpp>
#include <pxr/usd/sdf/path.h>

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace Cesium3DTilesSelection {
class RasterOverlay;
class Tileset;
class ViewState;
class ViewUpdateResult;
} // namespace Cesium3DTilesSelection

namespace CesiumGltf {
struct Model;
} // namespace CesiumGltf

namespace cesium::omniverse {
enum TilesetSourceType { ION = 0, URL = 1 };

class FabricPrepareRenderResources;
struct Viewport;

struct TilesetStatistics {
    uint64_t tilesetCachedBytes{0};
    uint64_t tilesVisited{0};
    uint64_t culledTilesVisited{0};
    uint64_t tilesRendered{0};
    uint64_t tilesCulled{0};
    uint64_t maxDepthVisited{0};
    uint64_t tilesLoadingWorker{0};
    uint64_t tilesLoadingMain{0};
    uint64_t tilesLoaded{0};
};

class OmniTileset {
  public:
    OmniTileset(const pxr::SdfPath& tilesetPath, const pxr::SdfPath& georeferencePath);
    ~OmniTileset();

    [[nodiscard]] pxr::SdfPath getPath() const;
    [[nodiscard]] std::string getName() const;
    [[nodiscard]] TilesetSourceType getSourceType() const;
    [[nodiscard]] std::string getUrl() const;
    [[nodiscard]] int64_t getIonAssetId() const;
    [[nodiscard]] std::optional<CesiumIonClient::Token> getIonAccessToken() const;
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
    [[nodiscard]] double getCulledScreenSpaceError() const;
    [[nodiscard]] bool getSuspendUpdate() const;
    [[nodiscard]] bool getSmoothNormals() const;
    [[nodiscard]] double getMainThreadLoadingTimeLimit() const;
    [[nodiscard]] bool getShowCreditsOnScreen() const;
    [[nodiscard]] pxr::CesiumGeoreference getGeoreference() const;
    [[nodiscard]] pxr::SdfPath getMaterialPath() const;

    [[nodiscard]] int64_t getTilesetId() const;
    [[nodiscard]] TilesetStatistics getStatistics() const;

    void updateTilesetOptionsFromProperties();

    void reload();
    void addImageryIon(const pxr::SdfPath& imageryPath);
    [[nodiscard]] std::optional<uint64_t>
    findImageryLayerIndex(const Cesium3DTilesSelection::RasterOverlay& overlay) const;
    [[nodiscard]] std::optional<uint64_t> findImageryLayerIndex(const pxr::SdfPath& imageryPath) const;
    [[nodiscard]] uint64_t getImageryLayerCount() const;
    [[nodiscard]] double getImageryLayerAlpha(uint64_t imageryLayerIndex) const;
    void updateImageryLayerAlpha(uint64_t imageryLayerIndex);
    void updateShaderInput(const pxr::SdfPath& shaderPath, const pxr::TfToken& attributeName);
    void onUpdateFrame(const std::vector<Viewport>& viewports);

  private:
    void updateTransform();
    void updateView(const std::vector<Viewport>& viewports);
    bool updateExtent();
    void updateLoadStatus();

    std::unique_ptr<Cesium3DTilesSelection::Tileset> _tileset;
    std::shared_ptr<FabricPrepareRenderResources> _renderResourcesPreparer;
    const Cesium3DTilesSelection::ViewUpdateResult* _pViewUpdateResult;

    pxr::SdfPath _tilesetPath;
    int64_t _tilesetId;
    glm::dmat4 _ecefToUsdTransform;
    std::vector<Cesium3DTilesSelection::ViewState> _viewStates;
    bool _extentSet = false;
    bool _activeLoading{false};
    std::vector<pxr::SdfPath> _imageryPaths;
};
} // namespace cesium::omniverse

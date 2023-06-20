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
struct ImageCesium;
class Tileset;
class ViewState;
class ViewUpdateResult;
} // namespace Cesium3DTilesSelection

namespace CesiumGltf {
struct ImageCesium;
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

    pxr::SdfPath getPath() const;
    std::string getName() const;
    TilesetSourceType getSourceType() const;
    std::string getUrl() const;
    int64_t getIonAssetId() const;
    std::optional<CesiumIonClient::Token> getIonAccessToken() const;
    float getMaximumScreenSpaceError() const;
    bool getPreloadAncestors() const;
    bool getPreloadSiblings() const;
    bool getForbidHoles() const;
    uint32_t getMaximumSimultaneousTileLoads() const;
    uint64_t getMaximumCachedBytes() const;
    uint32_t getLoadingDescendantLimit() const;
    bool getEnableFrustumCulling() const;
    bool getEnableFogCulling() const;
    bool getEnforceCulledScreenSpaceError() const;
    float getCulledScreenSpaceError() const;
    bool getSuspendUpdate() const;
    bool getSmoothNormals() const;
    bool getShowCreditsOnScreen() const;
    pxr::CesiumGeoreference getGeoreference() const;

    int64_t getTilesetId() const;
    TilesetStatistics getStatistics() const;

    void reload();
    void addImageryIon(const pxr::SdfPath& imageryPath);
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
};
} // namespace cesium::omniverse

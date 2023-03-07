#pragma once

#include <CesiumIonClient/Token.h>
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
class FabricPrepareRenderResources;

class OmniTileset {
  public:
    OmniTileset(const pxr::SdfPath& tilesetPath, int64_t tilesetId);
    ~OmniTileset();

    pxr::SdfPath getPath() const;
    std::string getName() const;
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

    int64_t getId() const;
    int64_t getNextTileId() const;

    void reload();
    void addImageryIon(const pxr::SdfPath& imageryPath);
    void onUpdateFrame(const std::vector<Cesium3DTilesSelection::ViewState>& viewStates);

  private:
    void updateTransform();
    void updateView(const std::vector<Cesium3DTilesSelection::ViewState>& viewStates);

    std::unique_ptr<Cesium3DTilesSelection::Tileset> _tileset;
    std::shared_ptr<FabricPrepareRenderResources> _renderResourcesPreparer;
    const Cesium3DTilesSelection::ViewUpdateResult* _pViewUpdateResult;

    pxr::SdfPath _tilesetPath;
    int64_t _tilesetId;
    mutable std::atomic<int64_t> _tileId{};
    glm::dmat4 _ecefToUsdTransform;
};
} // namespace cesium::omniverse

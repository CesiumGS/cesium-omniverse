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
    OmniTileset(int64_t tilesetId, const pxr::SdfPath& tilesetPath);
    ~OmniTileset();

    pxr::SdfPath getPath() const;
    std::string getName() const;
    std::string getUrl() const;
    int64_t getIonAssetId() const;
    std::optional<CesiumIonClient::Token> getIonToken() const;

    int64_t getId() const;
    int64_t getNextTileId() const;

    void reload();
    void addIonRasterOverlay(const pxr::SdfPath& rasterOverlayPath);
    void onUpdateFrame(const std::vector<Cesium3DTilesSelection::ViewState>& viewStates);

  private:
    void updateTransform();
    void updateView(const std::vector<Cesium3DTilesSelection::ViewState>& viewStates);

    // TODO: store this on the prim
    const bool _suspendUpdate = false;

    std::unique_ptr<Cesium3DTilesSelection::Tileset> _tileset;
    std::shared_ptr<FabricPrepareRenderResources> _renderResourcesPreparer;
    const Cesium3DTilesSelection::ViewUpdateResult* _pViewUpdateResult;

    pxr::SdfPath _tilesetPath;
    int64_t _tilesetId;
    mutable std::atomic<int64_t> _tileId{};
    glm::dmat4 _ecefToUsdTransform;
};
} // namespace cesium::omniverse

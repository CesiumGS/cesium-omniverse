#pragma once

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <Cesium3DTilesSelection/IPrepareRendererResources.h>
#include <pxr/usd/sdf/path.h>

namespace cesium::omniverse {

class OmniMesh;
class OmniTileset;

struct TileRenderResources {
    glm::dmat4 tileTransform;
    std::vector<std::shared_ptr<OmniMesh>> omniMeshes;
};

class OmniPrepareRenderResources : public Cesium3DTilesSelection::IPrepareRendererResources {
  public:
    OmniPrepareRenderResources(const OmniTileset& tileset);
    ~OmniPrepareRenderResources() = default;

    CesiumAsync::Future<Cesium3DTilesSelection::TileLoadResultAndRenderResources> prepareInLoadThread(
        const CesiumAsync::AsyncSystem& asyncSystem,
        Cesium3DTilesSelection::TileLoadResult&& tileLoadResult,
        const glm::dmat4& transform,
        const std::any& rendererOptions) override;

    void* prepareInMainThread(Cesium3DTilesSelection::Tile& tile, void* pLoadThreadResult) override;

    void free(Cesium3DTilesSelection::Tile& tile, void* pLoadThreadResult, void* pMainThreadResult) noexcept override;

    void* prepareRasterInLoadThread(CesiumGltf::ImageCesium& image, const std::any& rendererOptions) override;

    void*
    prepareRasterInMainThread(Cesium3DTilesSelection::RasterOverlayTile& rasterTile, void* pLoadThreadResult) override;

    void freeRaster(
        const Cesium3DTilesSelection::RasterOverlayTile& rasterTile,
        void* pLoadThreadResult,
        void* pMainThreadResult) noexcept override;

    void attachRasterInMainThread(
        const Cesium3DTilesSelection::Tile& tile,
        int32_t overlayTextureCoordinateID,
        const Cesium3DTilesSelection::RasterOverlayTile& rasterTile,
        void* pMainThreadRendererResources,
        const glm::dvec2& translation,
        const glm::dvec2& scale) override;

    void detachRasterInMainThread(
        const Cesium3DTilesSelection::Tile& tile,
        int32_t overlayTextureCoordinateID,
        const Cesium3DTilesSelection::RasterOverlayTile& rasterTile,
        void* pMainThreadRendererResources) noexcept override;

  private:
    const OmniTileset& _tileset;
};
} // namespace cesium::omniverse

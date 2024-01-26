#pragma once

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <Cesium3DTilesSelection/IPrepareRendererResources.h>

namespace cesium::omniverse {

class Context;
struct FabricMesh;
class OmniTileset;

class FabricPrepareRenderResources final : public Cesium3DTilesSelection::IPrepareRendererResources {
  public:
    FabricPrepareRenderResources(Context* pContext, OmniTileset* pTileset);
    ~FabricPrepareRenderResources() override = default;
    FabricPrepareRenderResources(const FabricPrepareRenderResources&) = delete;
    FabricPrepareRenderResources& operator=(const FabricPrepareRenderResources&) = delete;
    FabricPrepareRenderResources(FabricPrepareRenderResources&&) noexcept = default;
    FabricPrepareRenderResources& operator=(FabricPrepareRenderResources&&) noexcept = default;

    CesiumAsync::Future<Cesium3DTilesSelection::TileLoadResultAndRenderResources> prepareInLoadThread(
        const CesiumAsync::AsyncSystem& asyncSystem,
        Cesium3DTilesSelection::TileLoadResult&& tileLoadResult,
        const glm::dmat4& tileToEcefTransform,
        const std::any& rendererOptions) override;

    void* prepareInMainThread(Cesium3DTilesSelection::Tile& tile, void* pLoadThreadResult) override;

    void free(Cesium3DTilesSelection::Tile& tile, void* pLoadThreadResult, void* pMainThreadResult) noexcept override;

    void* prepareRasterInLoadThread(CesiumGltf::ImageCesium& image, const std::any& rendererOptions) override;

    void*
    prepareRasterInMainThread(CesiumRasterOverlays::RasterOverlayTile& rasterTile, void* pLoadThreadResult) override;

    void freeRaster(
        const CesiumRasterOverlays::RasterOverlayTile& rasterTile,
        void* pLoadThreadResult,
        void* pMainThreadResult) noexcept override;

    void attachRasterInMainThread(
        const Cesium3DTilesSelection::Tile& tile,
        int32_t overlayTextureCoordinateID,
        const CesiumRasterOverlays::RasterOverlayTile& rasterTile,
        void* pMainThreadRendererResources,
        const glm::dvec2& translation,
        const glm::dvec2& scale) override;

    void detachRasterInMainThread(
        const Cesium3DTilesSelection::Tile& tile,
        int32_t overlayTextureCoordinateID,
        const CesiumRasterOverlays::RasterOverlayTile& rasterTile,
        void* pMainThreadRendererResources) noexcept override;

    [[nodiscard]] bool tilesetExists() const;
    void detachTileset();

  private:
    Context* _pContext;
    OmniTileset* _pTileset;
};

} // namespace cesium::omniverse

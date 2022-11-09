#pragma once

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <Cesium3DTilesSelection/IPrepareRendererResources.h>
#include <CesiumGeometry/AxisTransforms.h>
#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usdGeom/xform.h>

namespace Cesium {
struct TileWorkerRenderResources {
    pxr::SdfLayerRefPtr layer;
    pxr::SdfPath primPath;
    bool enable;
};

struct TileRenderResources {
    pxr::UsdPrim prim;
    bool enable;
};

class RenderResourcesPreparer : public Cesium3DTilesSelection::IPrepareRendererResources {
  public:
    RenderResourcesPreparer(const pxr::UsdStageRefPtr& stage, const pxr::SdfPath& tilesetPath);

    void setTransform(const glm::dmat4& absToRelWorld);

    void setVisible(void* tileRenderResources, bool enable);

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

    pxr::UsdAttribute tilesetTransform;
    pxr::UsdStageRefPtr stage;
    pxr::SdfPath tilesetPath;
};
} // namespace Cesium

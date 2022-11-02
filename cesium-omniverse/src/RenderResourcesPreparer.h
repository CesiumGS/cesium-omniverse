#pragma once

#include <Cesium3DTilesSelection/IPrepareRendererResources.h>
#include <CesiumGeometry/AxisTransforms.h>

#ifdef CESIUM_OMNI_GCC
#define _GLIBCXX_PERMIT_BACKWARD_HASH
#endif

#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/xform.h>

#include "UtilMacros.h"

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
    RenderResourcesPreparer(const pxr::UsdStageRefPtr& stage_, const pxr::SdfPath& tilesetPath_)
        : stage{stage_}
        , tilesetPath{tilesetPath_} {
        auto xform = pxr::UsdGeomXform::Define(stage, tilesetPath_);
        const glm::dmat4& z_to_y = CesiumGeometry::AxisTransforms::Z_UP_TO_Y_UP;
        pxr::GfMatrix4d currentTransform{
            z_to_y[0][0],
            z_to_y[0][1],
            z_to_y[0][2],
            z_to_y[0][3],
            z_to_y[1][0],
            z_to_y[1][1],
            z_to_y[1][2],
            z_to_y[1][3],
            z_to_y[2][0],
            z_to_y[2][1],
            z_to_y[2][2],
            z_to_y[2][3],
            z_to_y[3][0],
            z_to_y[3][1],
            z_to_y[3][2],
            z_to_y[3][3],
        };
        tilesetTransform = xform.AddTransformOp();
        tilesetTransform.Set(currentTransform);
    }

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

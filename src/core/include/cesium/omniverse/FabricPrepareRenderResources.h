#pragma once

#include "cesium/omniverse/GltfUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <Cesium3DTilesSelection/IPrepareRendererResources.h>
#include <pxr/usd/sdf/path.h>

namespace cesium::omniverse {

class FabricGeometry;
class FabricMaterial;
class FabricTexture;
class OmniTileset;

struct FabricMesh {
    std::shared_ptr<FabricGeometry> geometry;
    std::shared_ptr<FabricMaterial> material;
    std::shared_ptr<FabricTexture> baseColorTexture;
    std::vector<std::shared_ptr<FabricTexture>> featureIdTextures;
    std::vector<std::shared_ptr<FabricTexture>> propertyTextures;
    MaterialInfo materialInfo;
    FeaturesInfo featuresInfo;
    std::unordered_map<uint64_t, uint64_t> texcoordIndexMapping;
    std::unordered_map<uint64_t, uint64_t> imageryTexcoordIndexMapping;
    std::vector<uint64_t> featureIdIndexSetIndexMapping;
    std::vector<uint64_t> featureIdAttributeSetIndexMapping;
    std::vector<uint64_t> featureIdTextureSetIndexMapping;
    std::unordered_map<uint64_t, uint64_t> propertyTextureIndexMapping;
};

struct TileRenderResources {
    glm::dmat4 tileTransform;
    std::vector<FabricMesh> fabricMeshes;
};

class FabricPrepareRenderResources final : public Cesium3DTilesSelection::IPrepareRendererResources {
  public:
    FabricPrepareRenderResources(const OmniTileset& tileset);
    ~FabricPrepareRenderResources() override = default;

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

    [[nodiscard]] bool tilesetExists() const;
    void detachTileset();

  private:
    const OmniTileset* _tileset;
};
} // namespace cesium::omniverse

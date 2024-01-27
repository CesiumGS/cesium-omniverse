#pragma once

#include "cesium/omniverse/FabricMaterialInfo.h"

#include <pxr/usd/usd/common.h>

#include <atomic>
#include <mutex>
#include <vector>

namespace CesiumGltf {
struct ImageCesium;
struct MeshPrimitive;
struct Model;
} // namespace CesiumGltf

namespace omni::ui {
class DynamicTextureProvider;
}

namespace cesium::omniverse {

class Context;
class FabricGeometry;
class FabricGeometryPool;
class FabricMaterial;
class FabricMaterialPool;
class FabricGeometryDescriptor;
class FabricMaterialDescriptor;
class FabricTexture;
class FabricTexturePool;
struct FabricFeaturesInfo;
struct FabricRasterOverlayLayersInfo;

class FabricResourceManager {
  public:
    FabricResourceManager(Context* pContext);
    ~FabricResourceManager();
    FabricResourceManager(const FabricResourceManager&) = delete;
    FabricResourceManager& operator=(const FabricResourceManager&) = delete;
    FabricResourceManager(FabricResourceManager&&) noexcept = delete;
    FabricResourceManager& operator=(FabricResourceManager&&) noexcept = delete;

    bool shouldAcquireMaterial(
        const CesiumGltf::MeshPrimitive& primitive,
        bool hasRasterOverlay,
        const pxr::SdfPath& tilesetMaterialPath) const;

    bool getDisableTextures() const;

    std::shared_ptr<FabricGeometry> acquireGeometry(
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        const FabricFeaturesInfo& featuresInfo,
        bool smoothNormals);

    std::shared_ptr<FabricMaterial> acquireMaterial(
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        const FabricMaterialInfo& materialInfo,
        const FabricFeaturesInfo& featuresInfo,
        const FabricRasterOverlayLayersInfo& rasterOverlayLayersInfo,
        int64_t tilesetId,
        const pxr::SdfPath& tilesetMaterialPath);

    std::shared_ptr<FabricTexture> acquireTexture();

    void releaseGeometry(std::shared_ptr<FabricGeometry> pGeometry);
    void releaseMaterial(std::shared_ptr<FabricMaterial> pMaterial);
    void releaseTexture(std::shared_ptr<FabricTexture> pTexture);

    void setDisableMaterials(bool disableMaterials);
    void setDisableTextures(bool disableTextures);
    void setDisableGeometryPool(bool disableGeometryPool);
    void setDisableMaterialPool(bool disableMaterialPool);
    void setDisableTexturePool(bool disableTexturePool);
    void setGeometryPoolInitialCapacity(uint64_t geometryPoolInitialCapacity);
    void setMaterialPoolInitialCapacity(uint64_t materialPoolInitialCapacity);
    void setTexturePoolInitialCapacity(uint64_t texturePoolInitialCapacity);
    void setDebugRandomColors(bool debugRandomColors);

    void updateShaderInput(
        const pxr::SdfPath& materialPath,
        const pxr::SdfPath& shaderPath,
        const pxr::TfToken& attributeName) const;

    void clear();

  private:
    struct SharedMaterial {
        SharedMaterial() = default;
        ~SharedMaterial() = default;
        SharedMaterial(const SharedMaterial&) = delete;
        SharedMaterial& operator=(const SharedMaterial&) = delete;
        SharedMaterial(SharedMaterial&&) noexcept = default;
        SharedMaterial& operator=(SharedMaterial&&) noexcept = default;

        std::shared_ptr<FabricMaterial> pMaterial;
        FabricMaterialInfo materialInfo;
        int64_t tilesetId;
        uint64_t referenceCount;
    };

    std::shared_ptr<FabricMaterial> createMaterial(const FabricMaterialDescriptor& materialDescriptor);

    std::shared_ptr<FabricMaterial> acquireSharedMaterial(
        const FabricMaterialInfo& materialInfo,
        const FabricMaterialDescriptor& materialDescriptor,
        int64_t tilesetId);
    void releaseSharedMaterial(const FabricMaterial& material);
    bool isSharedMaterial(const FabricMaterial& material) const;

    std::shared_ptr<FabricGeometry> acquireGeometryFromPool(const FabricGeometryDescriptor& geometryDescriptor);
    std::shared_ptr<FabricMaterial> acquireMaterialFromPool(const FabricMaterialDescriptor& materialDescriptor);
    std::shared_ptr<FabricTexture> acquireTextureFromPool();

    FabricGeometryPool* getGeometryPool(const FabricGeometry& geometry) const;
    FabricMaterialPool* getMaterialPool(const FabricMaterial& material) const;
    FabricTexturePool* getTexturePool(const FabricTexture& texture) const;

    int64_t getNextGeometryId();
    int64_t getNextMaterialId();
    int64_t getNextTextureId();
    int64_t getNextGeometryPoolId();
    int64_t getNextMaterialPoolId();
    int64_t getNextTexturePoolId();

    std::vector<std::unique_ptr<FabricGeometryPool>> _geometryPools;
    std::vector<std::unique_ptr<FabricMaterialPool>> _materialPools;
    std::vector<std::unique_ptr<FabricTexturePool>> _texturePools;

    bool _disableMaterials{false};
    bool _disableTextures{false};
    bool _disableGeometryPool{false};
    bool _disableMaterialPool{false};
    bool _disableTexturePool{false};

    uint64_t _geometryPoolInitialCapacity{0};
    uint64_t _materialPoolInitialCapacity{0};
    uint64_t _texturePoolInitialCapacity{0};

    bool _debugRandomColors{false};

    std::atomic<int64_t> _geometryId{0};
    std::atomic<int64_t> _materialId{0};
    std::atomic<int64_t> _textureId{0};
    std::atomic<int64_t> _geometryPoolId{0};
    std::atomic<int64_t> _materialPoolId{0};
    std::atomic<int64_t> _texturePoolId{0};

    std::mutex _poolMutex;

    Context* _pContext;
    std::unique_ptr<omni::ui::DynamicTextureProvider> _defaultWhiteTexture;
    std::unique_ptr<omni::ui::DynamicTextureProvider> _defaultTransparentTexture;
    pxr::TfToken _defaultWhiteTextureAssetPathToken;
    pxr::TfToken _defaultTransparentTextureAssetPathToken;

    std::vector<SharedMaterial> _sharedMaterials;
};

} // namespace cesium::omniverse

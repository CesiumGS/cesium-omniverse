#pragma once

#include "cesium/omniverse/GltfUtil.h"

#include <omni/fabric/IPath.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/path.h>

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

class FabricGeometry;
class FabricGeometryPool;
class FabricMaterial;
class FabricMaterialPool;
class FabricGeometryDefinition;
class FabricMaterialDefinition;
class FabricTexture;
class FabricTexturePool;

struct SharedMaterial {
    std::shared_ptr<FabricMaterial> material;
    MaterialInfo materialInfo;
    int64_t tilesetId;
    uint64_t referenceCount;
};

class FabricResourceManager {
  public:
    FabricResourceManager(const FabricResourceManager&) = delete;
    FabricResourceManager(FabricResourceManager&&) = delete;
    FabricResourceManager& operator=(const FabricResourceManager&) = delete;
    FabricResourceManager& operator=(FabricResourceManager) = delete;

    static FabricResourceManager& getInstance() {
        static FabricResourceManager instance;
        return instance;
    }

    bool shouldAcquireMaterial(
        const CesiumGltf::MeshPrimitive& primitive,
        bool hasImagery,
        const pxr::SdfPath& tilesetMaterialPath) const;

    std::shared_ptr<FabricGeometry> acquireGeometry(
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        bool smoothNormals,
        long stageId);

    std::shared_ptr<FabricMaterial> acquireMaterial(
        const MaterialInfo& materialInfo,
        bool hasImagery,
        const pxr::SdfPath& tilesetMaterialPath,
        long stageId,
        int64_t tilesetId);

    std::shared_ptr<FabricTexture> acquireTexture();

    void releaseGeometry(const std::shared_ptr<FabricGeometry>& geometry);
    void releaseMaterial(const std::shared_ptr<FabricMaterial>& material);
    void releaseTexture(const std::shared_ptr<FabricTexture>& texture);

    void setDisableMaterials(bool disableMaterials);
    void setDisableTextures(bool disableTextures);
    void setDisableGeometryPool(bool disableGeometryPool);
    void setDisableMaterialPool(bool disableMaterialPool);
    void setDisableTexturePool(bool disableTexturePool);
    void setGeometryPoolInitialCapacity(uint64_t geometryPoolInitialCapacity);
    void setMaterialPoolInitialCapacity(uint64_t materialPoolInitialCapacity);
    void setTexturePoolInitialCapacity(uint64_t texturePoolInitialCapacity);
    void setDebugRandomColors(bool debugRandomColors);

    void retainPath(const omni::fabric::Path& path);

    void clear();

  protected:
    FabricResourceManager();
    ~FabricResourceManager();

  private:
    std::shared_ptr<FabricMaterial> createMaterial(const FabricMaterialDefinition& materialDefinition, long stageId);

    void removeSharedMaterial(const SharedMaterial& sharedMaterial);
    SharedMaterial* getSharedMaterial(const MaterialInfo& materialInfo, int64_t tilesetId);
    SharedMaterial* getSharedMaterial(const std::shared_ptr<FabricMaterial>& material);
    std::shared_ptr<FabricMaterial> acquireSharedMaterial(
        const MaterialInfo& materialInfo,
        const FabricMaterialDefinition& materialDefinition,
        long stageId,
        int64_t tilesetId);
    void releaseSharedMaterial(const std::shared_ptr<FabricMaterial>& material);

    std::shared_ptr<FabricGeometryPool> getGeometryPool(const FabricGeometryDefinition& geometryDefinition);
    std::shared_ptr<FabricMaterialPool> getMaterialPool(const FabricMaterialDefinition& materialDefinition);
    std::shared_ptr<FabricTexturePool> getTexturePool();

    std::shared_ptr<FabricGeometryPool>
    createGeometryPool(const FabricGeometryDefinition& geometryDefinition, long stageId);
    std::shared_ptr<FabricMaterialPool>
    createMaterialPool(const FabricMaterialDefinition& materialDefinition, long stageId);
    std::shared_ptr<FabricTexturePool> createTexturePool();

    int64_t getNextGeometryId();
    int64_t getNextMaterialId();
    int64_t getNextTextureId();
    int64_t getNextPoolId();

    std::vector<std::shared_ptr<FabricGeometryPool>> _geometryPools;
    std::vector<std::shared_ptr<FabricMaterialPool>> _materialPools;
    std::vector<std::shared_ptr<FabricTexturePool>> _texturePools;

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
    std::atomic<int64_t> _poolId{0};

    std::mutex _poolMutex;

    std::unique_ptr<omni::ui::DynamicTextureProvider> _defaultTexture;
    pxr::TfToken _defaultTextureAssetPathToken;

    std::vector<omni::fabric::Path> _retainedPaths;

    std::vector<SharedMaterial> _sharedMaterials;
};

} // namespace cesium::omniverse

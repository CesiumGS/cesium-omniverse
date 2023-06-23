#pragma once

#include "cesium/omniverse/OmniMesh.h"

#include <mutex>

namespace CesiumGltf {
struct ImageCesium;
struct MeshPrimitive;
struct Model;
} // namespace CesiumGltf

namespace cesium::omniverse {

class OmniGeometryPool;
class OmniMaterialPool;
class OmniGeometryDefinition;
class OmniMaterialDefinition;

class OmniMeshManager {
  public:
    OmniMeshManager(const OmniMeshManager&) = delete;
    OmniMeshManager(OmniMeshManager&&) = delete;
    OmniMeshManager& operator=(const OmniMeshManager&) = delete;
    OmniMeshManager& operator=(OmniMeshManager) = delete;

    static OmniMeshManager& getInstance() {
        static OmniMeshManager instance;
        return instance;
    }

    std::shared_ptr<OmniMesh> acquireMesh(
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        bool smoothNormals,
        const CesiumGltf::ImageCesium* imagery,
        uint64_t imageryTexcoordSetIndex);

    void releaseMesh(std::shared_ptr<OmniMesh> mesh);

    void setDisableMaterials(bool disableMaterials);
    void setDisableTextures(bool disableTextures);
    void setDisableGeometryPool(bool disableGeometryPool);
    void setDisableMaterialPool(bool disableMaterialPool);
    void setGeometryPoolInitialCapacity(uint64_t geometryPoolInitialCapacity);
    void setMaterialPoolInitialCapacity(uint64_t materialPoolInitialCapacity);
    void setDebugRandomColors(bool debugRandomColors);

    void clear();

  protected:
    OmniMeshManager() = default;
    ~OmniMeshManager() = default;

  private:
    std::shared_ptr<OmniGeometry> acquireGeometry(
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        bool smoothNormals,
        const CesiumGltf::ImageCesium* imagery,
        uint64_t imageryTexcoordSetIndex);

    std::shared_ptr<OmniMaterial> acquireMaterial(
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        const CesiumGltf::ImageCesium* imagery);

    void releaseGeometry(std::shared_ptr<OmniGeometry> geometry);
    void releaseMaterial(std::shared_ptr<OmniMaterial> material);

    std::shared_ptr<OmniGeometryPool> getGeometryPool(const OmniGeometryDefinition& geometryDefinition);
    std::shared_ptr<OmniMaterialPool> getMaterialPool(const OmniMaterialDefinition& materialDefinition);

    int64_t getNextGeometryId();
    int64_t getNextMaterialId();
    int64_t getNextPoolId();

    std::vector<std::shared_ptr<OmniGeometryPool>> _geometryPools;
    std::vector<std::shared_ptr<OmniMaterialPool>> _materialPools;

    bool _disableMaterials{false};
    bool _disableTextures{false};
    bool _disableGeometryPool{false};
    bool _disableMaterialPool{false};

    uint64_t _geometryPoolInitialCapacity{0};
    uint64_t _materialPoolInitialCapacity{0};

    bool _debugRandomColors{false};

    std::atomic<int64_t> _geometryId{0};
    std::atomic<int64_t> _materialId{0};
    std::atomic<int64_t> _poolId{0};

    std::mutex _poolMutex;
};

} // namespace cesium::omniverse

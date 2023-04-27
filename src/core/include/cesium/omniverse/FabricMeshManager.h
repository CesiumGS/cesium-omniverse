#pragma once

#include "cesium/omniverse/FabricMesh.h"

#include <mutex>

namespace CesiumGltf {
struct ImageCesium;
struct MeshPrimitive;
struct Model;
} // namespace CesiumGltf

namespace cesium::omniverse {

class FabricGeometryPool;
class FabricMaterialPool;
class FabricGeometryDefinition;
class FabricMaterialDefinition;

class FabricMeshManager {
  public:
    FabricMeshManager(const FabricMeshManager&) = delete;
    FabricMeshManager(FabricMeshManager&&) = delete;
    FabricMeshManager& operator=(const FabricMeshManager&) = delete;
    FabricMeshManager& operator=(FabricMeshManager) = delete;

    static FabricMeshManager& getInstance() {
        static FabricMeshManager instance;
        return instance;
    }

    std::shared_ptr<FabricMesh> FabricMeshManager::acquireMesh(
        int64_t tilesetId,
        int64_t tileId,
        const glm::dmat4& ecefToUsdTransform,
        const glm::dmat4& gltfToEcefTransform,
        const glm::dmat4& nodeTransform,
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        bool smoothNormals,
        const CesiumGltf::ImageCesium* imagery,
        const glm::dvec2& imageryTexcoordTranslation,
        const glm::dvec2& imageryTexcoordScale,
        uint64_t imageryTexcoordSetIndex);

    void releaseMesh(std::shared_ptr<FabricMesh> mesh);

    uint64_t getNumberOfGeometriesInUse() const;
    uint64_t getNumberOfMaterialsInUse() const;

    uint64_t getGeometryPoolCapacity() const;
    uint64_t getMaterialPoolCapacity() const;

    void clear();

  protected:
    FabricMeshManager() = default;
    ~FabricMeshManager() = default;

  private:
    std::shared_ptr<FabricGeometry> acquireGeometry(
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        bool smoothNormals,
        const CesiumGltf::ImageCesium* imagery,
        uint64_t imageryTexcoordSetIndex);

    std::shared_ptr<FabricMaterial> acquireMaterial(
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        const CesiumGltf::ImageCesium* imagery);

    void releaseGeometry(std::shared_ptr<FabricGeometry> geometry);
    void releaseMaterial(std::shared_ptr<FabricMaterial> material);

    std::shared_ptr<FabricGeometryPool> getGeometryPool(const FabricGeometryDefinition& geometryDefinition);
    std::shared_ptr<FabricMaterialPool> getMaterialPool(const FabricMaterialDefinition& materialDefinition);

    int64_t getNextGeometryId();
    int64_t getNextMaterialId();
    int64_t getNextPoolId();

    std::vector<std::shared_ptr<FabricGeometryPool>> _geometryPools;
    std::vector<std::shared_ptr<FabricMaterialPool>> _materialPools;

    bool _debugDisableGeometryPool{true};
    bool _debugDisableMaterialPool{false};

    std::atomic<int64_t> _geometryId{0};
    std::atomic<int64_t> _materialId{0};
    std::atomic<int64_t> _poolId{0};

    std::mutex _poolMutex;
};

} // namespace cesium::omniverse

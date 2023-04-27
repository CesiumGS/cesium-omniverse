#include "cesium/omniverse/FabricMeshManager.h"

#include "cesium/omniverse/FabricGeometry.h"
#include "cesium/omniverse/FabricGeometryDefinition.h"
#include "cesium/omniverse/FabricGeometryPool.h"
#include "cesium/omniverse/FabricMaterial.h"
#include "cesium/omniverse/FabricMaterialDefinition.h"
#include "cesium/omniverse/FabricMaterialPool.h"
#include "cesium/omniverse/GltfUtil.h"

#include <spdlog/fmt/fmt.h>

namespace cesium::omniverse {

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
    uint64_t imageryTexcoordSetIndex) {

    const auto hasImagery = imagery != nullptr;
    const auto geometry = acquireGeometry(model, primitive, smoothNormals, imagery, imageryTexcoordSetIndex);

    geometry->setTile(
        tilesetId,
        tileId,
        ecefToUsdTransform,
        gltfToEcefTransform,
        nodeTransform,
        model,
        primitive,
        smoothNormals,
        hasImagery,
        imageryTexcoordTranslation,
        imageryTexcoordScale,
        imageryTexcoordSetIndex);

    const auto hasMaterial = geometry->getGeometryDefinition().hasMaterial();
    auto material = std::shared_ptr<FabricMaterial>(nullptr);

    if (hasMaterial) {
        material = acquireMaterial(model, primitive, imagery);

        material->setTile(tilesetId, tileId, model, primitive, imagery);

        geometry->assignMaterial(material);
    }

    return std::make_shared<FabricMesh>(geometry, material);
}

void FabricMeshManager::releaseMesh(std::shared_ptr<FabricMesh> mesh) {
    const auto geometry = mesh->getGeometry();
    const auto material = mesh->getMaterial();

    assert(geometry != nullptr);

    releaseGeometry(geometry);

    if (material != nullptr) {
        releaseMaterial(material);
    }
}

uint64_t FabricMeshManager::getNumberOfGeometriesInUse() const {
    uint64_t count = 0;
    for (const auto geometryPool : _geometryPools) {
        count += geometryPool->getNumberActive();
    }
    return count;
}

uint64_t FabricMeshManager::getNumberOfMaterialsInUse() const {
    uint64_t count = 0;
    for (const auto materialPool : _materialPools) {
        count += materialPool->getNumberActive();
    }
    return count;
}

uint64_t FabricMeshManager::getGeometryPoolCapacity() const {
    uint64_t count = 0;
    for (const auto geometryPool : _geometryPools) {
        count += geometryPool->getCapacity();
    }
    return count;
}

uint64_t FabricMeshManager::getMaterialPoolCapacity() const {
    uint64_t count = 0;
    for (const auto materialPool : _materialPools) {
        count += materialPool->getCapacity();
    }
    return count;
}

void FabricMeshManager::clear() {
    _geometryPools.clear();
    _materialPools.clear();
}

std::shared_ptr<FabricGeometry> FabricMeshManager::acquireGeometry(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    bool smoothNormals,
    const CesiumGltf::ImageCesium* imagery,
    uint64_t imageryTexcoordSetIndex) {

    const auto hasImagery = imagery != nullptr;
    FabricGeometryDefinition geometryDefinition(model, primitive, smoothNormals, hasImagery, imageryTexcoordSetIndex);

    if (_debugDisableGeometryPool) {
        const auto path = pxr::SdfPath(fmt::format("/fabric_geometry_{}", getNextGeometryId()));
        return std::make_shared<FabricGeometry>(path, geometryDefinition);
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    const auto geometryPool = getGeometryPool(geometryDefinition);
    const auto geometry = geometryPool->acquire();

    return geometry;
}
std::shared_ptr<FabricMaterial> FabricMeshManager::acquireMaterial(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const CesiumGltf::ImageCesium* imagery) {

    const auto hasImagery = imagery != nullptr;
    FabricMaterialDefinition materialDefinition(model, primitive, hasImagery);

    if (_debugDisableMaterialPool) {
        const auto path = pxr::SdfPath(fmt::format("/fabric_material_{}", getNextMaterialId()));
        return std::make_shared<FabricMaterial>(path, materialDefinition);
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    const auto materialPool = getMaterialPool(materialDefinition);
    const auto material = materialPool->acquire();

    return material;
}

void FabricMeshManager::releaseGeometry(std::shared_ptr<FabricGeometry> geometry) {
    if (_debugDisableGeometryPool) {
        return;
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    const auto geometryPool = getGeometryPool(geometry->getGeometryDefinition());
    geometryPool->release(geometry);
}

void FabricMeshManager::releaseMaterial(std::shared_ptr<FabricMaterial> material) {
    if (_debugDisableMaterialPool) {
        return;
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    const auto materialPool = getMaterialPool(material->getMaterialDefinition());
    materialPool->release(material);
}

std::shared_ptr<FabricGeometryPool>
FabricMeshManager::getGeometryPool(const FabricGeometryDefinition& geometryDefinition) {
    for (const auto geometryPool : _geometryPools) {
        if (geometryDefinition == geometryPool->getGeometryDefinition()) {
            // Found a pool with the same geometry definition
            return geometryPool;
        }
    }

    // Create a new pool
    return _geometryPools.emplace_back(std::make_shared<FabricGeometryPool>(getNextPoolId(), geometryDefinition));
}

std::shared_ptr<FabricMaterialPool>
FabricMeshManager::getMaterialPool(const FabricMaterialDefinition& materialDefinition) {
    for (const auto materialPool : _materialPools) {
        if (materialDefinition == materialPool->getMaterialDefinition()) {
            // Found a pool with the same material definition
            return materialPool;
        }
    }

    // Create a new pool
    return _materialPools.emplace_back(std::make_shared<FabricMaterialPool>(getNextPoolId(), materialDefinition));
}

int64_t FabricMeshManager::getNextGeometryId() {
    return _geometryId++;
}

int64_t FabricMeshManager::getNextMaterialId() {
    return _materialId++;
}

int64_t FabricMeshManager::getNextPoolId() {
    return _poolId++;
}

}; // namespace cesium::omniverse

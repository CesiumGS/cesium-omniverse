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
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    bool smoothNormals,
    const CesiumGltf::ImageCesium* imagery,
    uint64_t imageryTexcoordSetIndex) {

    const auto geometry = acquireGeometry(model, primitive, smoothNormals, imagery, imageryTexcoordSetIndex);

    const auto hasMaterial = geometry->getGeometryDefinition().hasMaterial();
    auto material = std::shared_ptr<FabricMaterial>(nullptr);

    if (hasMaterial) {
        material = acquireMaterial(model, primitive, imagery);
    }

    return std::make_shared<FabricMesh>(geometry, material);
}

void FabricMeshManager::releaseMesh(const std::shared_ptr<FabricMesh>& mesh) {
    const auto geometry = mesh->getGeometry();
    const auto material = mesh->getMaterial();

    assert(geometry != nullptr);

    releaseGeometry(geometry);

    if (material != nullptr) {
        releaseMaterial(material);
    }
}

void FabricMeshManager::setDisableMaterials(bool disableMaterials) {
    _disableMaterials = disableMaterials;
}

void FabricMeshManager::setDisableTextures(bool disableTextures) {
    _disableTextures = disableTextures;
}

void FabricMeshManager::setDisableGeometryPool(bool disableGeometryPool) {
    assert(_geometryPools.size() == 0);
    _disableGeometryPool = disableGeometryPool;
}

void FabricMeshManager::setDisableMaterialPool(bool disableMaterialPool) {
    assert(_materialPools.size() == 0);
    _disableMaterialPool = disableMaterialPool;
}

void FabricMeshManager::setGeometryPoolInitialCapacity(uint64_t geometryPoolInitialCapacity) {
    assert(_geometryPools.size() == 0);
    _geometryPoolInitialCapacity = geometryPoolInitialCapacity;
}

void FabricMeshManager::setMaterialPoolInitialCapacity(uint64_t materialPoolInitialCapacity) {
    assert(_materialPools.size() == 0);
    _materialPoolInitialCapacity = materialPoolInitialCapacity;
}

void FabricMeshManager::setDebugRandomColors(bool debugRandomColors) {
    _debugRandomColors = debugRandomColors;
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
    FabricGeometryDefinition geometryDefinition(
        model, primitive, smoothNormals, hasImagery, imageryTexcoordSetIndex, _disableMaterials);

    if (_disableGeometryPool) {
        const auto path = pxr::SdfPath(fmt::format("/fabric_geometry_{}", getNextGeometryId()));
        return std::make_shared<FabricGeometry>(path, geometryDefinition, _debugRandomColors);
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    const auto geometryPool = getGeometryPool(geometryDefinition);
    auto geometry = geometryPool->acquire();

    return geometry;
}
std::shared_ptr<FabricMaterial> FabricMeshManager::acquireMaterial(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const CesiumGltf::ImageCesium* imagery) {

    const auto hasImagery = imagery != nullptr;
    FabricMaterialDefinition materialDefinition(model, primitive, hasImagery, _disableTextures);

    if (_disableMaterialPool) {
        const auto path = pxr::SdfPath(fmt::format("/fabric_material_{}", getNextMaterialId()));
        return std::make_shared<FabricMaterial>(path, materialDefinition);
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    const auto materialPool = getMaterialPool(materialDefinition);
    auto material = materialPool->acquire();

    return material;
}

void FabricMeshManager::releaseGeometry(const std::shared_ptr<FabricGeometry>& geometry) {
    if (_disableGeometryPool) {
        return;
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    const auto geometryPool = getGeometryPool(geometry->getGeometryDefinition());
    geometryPool->release(geometry);
}

void FabricMeshManager::releaseMaterial(const std::shared_ptr<FabricMaterial>& material) {
    if (_disableMaterialPool) {
        return;
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    const auto materialPool = getMaterialPool(material->getMaterialDefinition());
    materialPool->release(material);
}

std::shared_ptr<FabricGeometryPool>
FabricMeshManager::getGeometryPool(const FabricGeometryDefinition& geometryDefinition) {
    for (const auto& geometryPool : _geometryPools) {
        if (geometryDefinition == geometryPool->getGeometryDefinition()) {
            // Found a pool with the same geometry definition
            return geometryPool;
        }
    }

    // Create a new pool
    return _geometryPools.emplace_back(std::make_shared<FabricGeometryPool>(
        getNextPoolId(), geometryDefinition, _geometryPoolInitialCapacity, _debugRandomColors));
}

std::shared_ptr<FabricMaterialPool>
FabricMeshManager::getMaterialPool(const FabricMaterialDefinition& materialDefinition) {
    for (const auto& materialPool : _materialPools) {
        if (materialDefinition == materialPool->getMaterialDefinition()) {
            // Found a pool with the same material definition
            return materialPool;
        }
    }

    // Create a new pool
    return _materialPools.emplace_back(
        std::make_shared<FabricMaterialPool>(getNextPoolId(), materialDefinition, _materialPoolInitialCapacity));
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

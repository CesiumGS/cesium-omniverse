#include "cesium/omniverse/OmniMeshManager.h"

#include "cesium/omniverse/FabricGeometry.h"
#include "cesium/omniverse/FabricMaterial.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/OmniGeometry.h"
#include "cesium/omniverse/OmniGeometryDefinition.h"
#include "cesium/omniverse/OmniGeometryPool.h"
#include "cesium/omniverse/OmniMaterial.h"
#include "cesium/omniverse/OmniMaterialDefinition.h"
#include "cesium/omniverse/OmniMaterialPool.h"

#include <spdlog/fmt/fmt.h>

namespace cesium::omniverse {

std::shared_ptr<OmniMesh> OmniMeshManager::acquireMesh(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    bool smoothNormals,
    const CesiumGltf::ImageCesium* imagery,
    uint64_t imageryTexcoordSetIndex) {

    const auto geometry = acquireGeometry(model, primitive, smoothNormals, imagery, imageryTexcoordSetIndex);

    const auto hasMaterial = geometry->getGeometryDefinition().hasMaterial();
    auto material = std::shared_ptr<OmniMaterial>(nullptr);

    if (hasMaterial) {
        material = acquireMaterial(model, primitive, imagery);
    }

    return std::make_shared<OmniMesh>(geometry, material);
}

void OmniMeshManager::releaseMesh(std::shared_ptr<OmniMesh> mesh) {
    const auto geometry = mesh->getGeometry();
    const auto material = mesh->getMaterial();

    assert(geometry != nullptr);

    releaseGeometry(geometry);

    if (material != nullptr) {
        releaseMaterial(material);
    }
}

void OmniMeshManager::setSceneDelegate(OmniSceneDelegate sceneDelegate) {
    assert(_geometryPools.size() == 0);
    _sceneDelegate = sceneDelegate;
}

void OmniMeshManager::setDisableMaterials(bool disableMaterials) {
    assert(_geometryPools.size() == 0);
    _disableMaterials = disableMaterials;
}

void OmniMeshManager::setDisableTextures(bool disableTextures) {
    assert(_geometryPools.size() == 0);
    _disableTextures = disableTextures;
}

void OmniMeshManager::setDisableGeometryPool(bool disableGeometryPool) {
    assert(_geometryPools.size() == 0);
    _disableGeometryPool = disableGeometryPool;
}

void OmniMeshManager::setDisableMaterialPool(bool disableMaterialPool) {
    assert(_materialPools.size() == 0);
    _disableMaterialPool = disableMaterialPool;
}

void OmniMeshManager::setGeometryPoolInitialCapacity(uint64_t geometryPoolInitialCapacity) {
    assert(_geometryPools.size() == 0);
    _geometryPoolInitialCapacity = geometryPoolInitialCapacity;
}

void OmniMeshManager::setMaterialPoolInitialCapacity(uint64_t materialPoolInitialCapacity) {
    assert(_materialPools.size() == 0);
    _materialPoolInitialCapacity = materialPoolInitialCapacity;
}

void OmniMeshManager::setDebugRandomColors(bool debugRandomColors) {
    assert(_geometryPools.size() == 0);
    _debugRandomColors = debugRandomColors;
}

void OmniMeshManager::clear() {
    _geometryPools.clear();
    _materialPools.clear();
}

std::shared_ptr<OmniGeometry> OmniMeshManager::acquireGeometry(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    bool smoothNormals,
    const CesiumGltf::ImageCesium* imagery,
    uint64_t imageryTexcoordSetIndex) {

    const auto hasImagery = imagery != nullptr;
    OmniGeometryDefinition geometryDefinition(
        model, primitive, smoothNormals, hasImagery, imageryTexcoordSetIndex, _disableMaterials);

    if (_disableGeometryPool) {
        const auto path = pxr::SdfPath(fmt::format("/omni_geometry_{}", getNextGeometryId()));
        return std::make_shared<FabricGeometry>(path, geometryDefinition, _debugRandomColors);
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    const auto geometryPool = getGeometryPool(geometryDefinition);
    const auto geometry = geometryPool->acquire();

    return geometry;
}
std::shared_ptr<OmniMaterial> OmniMeshManager::acquireMaterial(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const CesiumGltf::ImageCesium* imagery) {

    const auto hasImagery = imagery != nullptr;
    OmniMaterialDefinition materialDefinition(model, primitive, hasImagery, _disableTextures);

    if (_disableMaterialPool) {
        const auto path = pxr::SdfPath(fmt::format("/omni_material_{}", getNextMaterialId()));
        return std::make_shared<FabricMaterial>(path, materialDefinition);
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    const auto materialPool = getMaterialPool(materialDefinition);
    const auto material = materialPool->acquire();

    return material;
}

void OmniMeshManager::releaseGeometry(std::shared_ptr<OmniGeometry> geometry) {
    if (_disableGeometryPool) {
        return;
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    const auto geometryPool = getGeometryPool(geometry->getGeometryDefinition());
    geometryPool->release(geometry);
}

void OmniMeshManager::releaseMaterial(std::shared_ptr<OmniMaterial> material) {
    if (_disableMaterialPool) {
        return;
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    const auto materialPool = getMaterialPool(material->getMaterialDefinition());
    materialPool->release(material);
}

std::shared_ptr<OmniGeometryPool> OmniMeshManager::getGeometryPool(const OmniGeometryDefinition& geometryDefinition) {
    for (const auto& geometryPool : _geometryPools) {
        if (geometryDefinition == geometryPool->getGeometryDefinition()) {
            // Found a pool with the same geometry definition
            return geometryPool;
        }
    }

    // Create a new pool
    return _geometryPools.emplace_back(std::make_shared<OmniGeometryPool>(
        getNextPoolId(), geometryDefinition, _geometryPoolInitialCapacity, _debugRandomColors, _sceneDelegate));
}

std::shared_ptr<OmniMaterialPool> OmniMeshManager::getMaterialPool(const OmniMaterialDefinition& materialDefinition) {
    for (const auto& materialPool : _materialPools) {
        if (materialDefinition == materialPool->getMaterialDefinition()) {
            // Found a pool with the same material definition
            return materialPool;
        }
    }

    // Create a new pool
    return _materialPools.emplace_back(std::make_shared<OmniMaterialPool>(
        getNextPoolId(), materialDefinition, _materialPoolInitialCapacity, _sceneDelegate));
}

int64_t OmniMeshManager::getNextGeometryId() {
    return _geometryId++;
}

int64_t OmniMeshManager::getNextMaterialId() {
    return _materialId++;
}

int64_t OmniMeshManager::getNextPoolId() {
    return _poolId++;
}

}; // namespace cesium::omniverse

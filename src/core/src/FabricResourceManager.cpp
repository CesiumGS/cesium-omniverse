#include "cesium/omniverse/FabricResourceManager.h"

#include "cesium/omniverse/FabricGeometry.h"
#include "cesium/omniverse/FabricGeometryDefinition.h"
#include "cesium/omniverse/FabricGeometryPool.h"
#include "cesium/omniverse/FabricMaterial.h"
#include "cesium/omniverse/FabricMaterialDefinition.h"
#include "cesium/omniverse/FabricMaterialPool.h"
#include "cesium/omniverse/FabricTexture.h"
#include "cesium/omniverse/FabricTexturePool.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/UsdUtil.h"

#include <omni/ui/ImageProvider/DynamicTextureProvider.h>
#include <spdlog/fmt/fmt.h>

namespace cesium::omniverse {

FabricResourceManager::FabricResourceManager() {
    const auto defaultTextureName = "fabric_default_texture";
    _defaultTextureAssetPath = UsdUtil::getDynamicTextureProviderAssetPath(defaultTextureName);
    _defaultTexture = std::make_unique<omni::ui::DynamicTextureProvider>(defaultTextureName);

    const auto bytes = std::array<uint8_t, 4>{{255, 255, 255, 255}};
    const auto size = carb::Uint2{1, 1};
    _defaultTexture->setBytesData(bytes.data(), size, omni::ui::kAutoCalculateStride, carb::Format::eRGBA8_SRGB);
}

FabricResourceManager::~FabricResourceManager() = default;

std::shared_ptr<FabricGeometry> FabricResourceManager::acquireGeometry(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    bool smoothNormals,
    bool hasImagery) {

    FabricGeometryDefinition geometryDefinition(model, primitive, smoothNormals, hasImagery, _disableMaterials);

    if (_disableGeometryPool) {
        const auto path = pxr::SdfPath(fmt::format("/fabric_geometry_{}", getNextGeometryId()));
        return std::make_shared<FabricGeometry>(path, geometryDefinition, _debugRandomColors);
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    const auto geometryPool = getGeometryPool(geometryDefinition);
    auto geometry = geometryPool->acquire();

    return geometry;
}
std::shared_ptr<FabricMaterial> FabricResourceManager::acquireMaterial(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    bool hasImagery) {

    FabricMaterialDefinition materialDefinition(model, primitive, hasImagery, _disableTextures);

    if (_disableMaterialPool) {
        const auto path = pxr::SdfPath(fmt::format("/fabric_material_{}", getNextMaterialId()));
        return std::make_shared<FabricMaterial>(path, materialDefinition, _defaultTextureAssetPath);
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    const auto materialPool = getMaterialPool(materialDefinition);
    auto material = materialPool->acquire();

    return material;
}

std::shared_ptr<FabricTexture> FabricResourceManager::acquireTexture() {
    if (_disableTexturePool) {
        const auto name = fmt::format("/fabric_texture_{}", getNextTextureId());
        return std::make_shared<FabricTexture>(name);
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    const auto texturePool = getTexturePool();
    auto texture = texturePool->acquire();

    return texture;
}

void FabricResourceManager::releaseGeometry(const std::shared_ptr<FabricGeometry>& geometry) {
    if (_disableGeometryPool) {
        return;
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    const auto geometryPool = getGeometryPool(geometry->getGeometryDefinition());
    geometryPool->release(geometry);
}

void FabricResourceManager::releaseMaterial(const std::shared_ptr<FabricMaterial>& material) {
    if (_disableMaterialPool) {
        return;
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    const auto materialPool = getMaterialPool(material->getMaterialDefinition());
    materialPool->release(material);
}

void FabricResourceManager::releaseTexture(const std::shared_ptr<FabricTexture>& texture) {
    if (_disableTexturePool) {
        return;
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    const auto texturePool = getTexturePool();
    texturePool->release(texture);
}

void FabricResourceManager::setDisableMaterials(bool disableMaterials) {
    _disableMaterials = disableMaterials;
}

void FabricResourceManager::setDisableTextures(bool disableTextures) {
    _disableTextures = disableTextures;
}

void FabricResourceManager::setDisableGeometryPool(bool disableGeometryPool) {
    assert(_geometryPools.size() == 0);
    _disableGeometryPool = disableGeometryPool;
}

void FabricResourceManager::setDisableMaterialPool(bool disableMaterialPool) {
    assert(_materialPools.size() == 0);
    _disableMaterialPool = disableMaterialPool;
}

void FabricResourceManager::setDisableTexturePool(bool disableTexturePool) {
    assert(_texturePools.size() == 0);
    _disableTexturePool = disableTexturePool;
}

void FabricResourceManager::setGeometryPoolInitialCapacity(uint64_t geometryPoolInitialCapacity) {
    assert(_geometryPools.size() == 0);
    _geometryPoolInitialCapacity = geometryPoolInitialCapacity;
}

void FabricResourceManager::setMaterialPoolInitialCapacity(uint64_t materialPoolInitialCapacity) {
    assert(_materialPools.size() == 0);
    _materialPoolInitialCapacity = materialPoolInitialCapacity;
}

void FabricResourceManager::setTexturePoolInitialCapacity(uint64_t texturePoolInitialCapacity) {
    assert(_texturePools.size() == 0);
    _texturePoolInitialCapacity = texturePoolInitialCapacity;
}

void FabricResourceManager::setDebugRandomColors(bool debugRandomColors) {
    _debugRandomColors = debugRandomColors;
}

void FabricResourceManager::clear() {
    _geometryPools.clear();
    _materialPools.clear();
    _texturePools.clear();
}

std::shared_ptr<FabricGeometryPool>
FabricResourceManager::getGeometryPool(const FabricGeometryDefinition& geometryDefinition) {
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
FabricResourceManager::getMaterialPool(const FabricMaterialDefinition& materialDefinition) {
    for (const auto& materialPool : _materialPools) {
        if (materialDefinition == materialPool->getMaterialDefinition()) {
            // Found a pool with the same material definition
            return materialPool;
        }
    }

    // Create a new pool
    return _materialPools.emplace_back(std::make_shared<FabricMaterialPool>(
        getNextPoolId(), materialDefinition, _materialPoolInitialCapacity, _defaultTextureAssetPath));
}

std::shared_ptr<FabricTexturePool> FabricResourceManager::getTexturePool() {
    if (!_texturePools.empty()) {
        return _texturePools.front();
    }

    // Create a new pool
    return _texturePools.emplace_back(
        std::make_shared<FabricTexturePool>(getNextPoolId(), _texturePoolInitialCapacity));
}

int64_t FabricResourceManager::getNextGeometryId() {
    return _geometryId++;
}

int64_t FabricResourceManager::getNextMaterialId() {
    return _materialId++;
}

int64_t FabricResourceManager::getNextTextureId() {
    return _textureId++;
}

int64_t FabricResourceManager::getNextPoolId() {
    return _poolId++;
}

}; // namespace cesium::omniverse

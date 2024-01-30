#include "cesium/omniverse/FabricResourceManager.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/FabricGeometry.h"
#include "cesium/omniverse/FabricGeometryDescriptor.h"
#include "cesium/omniverse/FabricGeometryPool.h"
#include "cesium/omniverse/FabricMaterialDescriptor.h"
#include "cesium/omniverse/FabricMaterialInfo.h"
#include "cesium/omniverse/FabricMaterialPool.h"
#include "cesium/omniverse/FabricPropertyDescriptor.h"
#include "cesium/omniverse/FabricTexture.h"
#include "cesium/omniverse/FabricTexturePool.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/FabricVertexAttributeDescriptor.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/MetadataUtil.h"
#include "cesium/omniverse/UsdUtil.h"

#include <omni/ui/ImageProvider/DynamicTextureProvider.h>
#include <spdlog/fmt/fmt.h>

namespace cesium::omniverse {

namespace {

const std::string_view DEFAULT_WHITE_TEXTURE_NAME = "fabric_default_white_texture";
const std::string_view DEFAULT_TRANSPARENT_TEXTURE_NAME = "fabric_default_transparent_texture";

std::unique_ptr<omni::ui::DynamicTextureProvider>
createSinglePixelTexture(const std::string_view& name, const std::array<uint8_t, 4>& bytes) {
    const auto size = carb::Uint2{1, 1};
    auto pTexture = std::make_unique<omni::ui::DynamicTextureProvider>(std::string(name));
    pTexture->setBytesData(bytes.data(), size, omni::ui::kAutoCalculateStride, carb::Format::eRGBA8_SRGB);
    return pTexture;
}

bool shouldAcquireSharedMaterial(const FabricMaterialDescriptor& materialDescriptor) {
    if (materialDescriptor.hasBaseColorTexture() || materialDescriptor.getRasterOverlayRenderMethods().size() > 0 ||
        !materialDescriptor.getFeatureIdTypes().empty() || !materialDescriptor.getStyleableProperties().empty()) {
        return false;
    }

    return true;
}

} // namespace

FabricResourceManager::FabricResourceManager(Context* pContext)
    : _pContext(pContext)
    , _defaultWhiteTexture(createSinglePixelTexture(DEFAULT_WHITE_TEXTURE_NAME, {{255, 255, 255, 255}}))
    , _defaultTransparentTexture(createSinglePixelTexture(DEFAULT_TRANSPARENT_TEXTURE_NAME, {{0, 0, 0, 0}}))
    , _defaultWhiteTextureAssetPathToken(UsdUtil::getDynamicTextureProviderAssetPathToken(DEFAULT_WHITE_TEXTURE_NAME))
    , _defaultTransparentTextureAssetPathToken(
          UsdUtil::getDynamicTextureProviderAssetPathToken(DEFAULT_TRANSPARENT_TEXTURE_NAME))

{}

FabricResourceManager::~FabricResourceManager() = default;

bool FabricResourceManager::shouldAcquireMaterial(
    const CesiumGltf::MeshPrimitive& primitive,
    bool hasRasterOverlay,
    const pxr::SdfPath& tilesetMaterialPath) const {
    if (_disableMaterials) {
        return false;
    }

    if (!tilesetMaterialPath.IsEmpty()) {
        return FabricUtil::materialHasCesiumNodes(
            _pContext->getFabricStage(), FabricUtil::toFabricPath(tilesetMaterialPath));
    }

    return hasRasterOverlay || GltfUtil::hasMaterial(primitive);
}

bool FabricResourceManager::getDisableTextures() const {
    return _disableTextures;
}

std::shared_ptr<FabricGeometry> FabricResourceManager::acquireGeometry(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const FabricFeaturesInfo& featuresInfo,
    bool smoothNormals) {

    FabricGeometryDescriptor geometryDescriptor(model, primitive, featuresInfo, smoothNormals);

    if (_disableGeometryPool) {
        const auto pathStr = fmt::format("/cesium_geometry_{}", getNextGeometryId());
        const auto path = omni::fabric::Path(pathStr.c_str());
        return std::make_shared<FabricGeometry>(_pContext, path, geometryDescriptor, -1);
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    return acquireGeometryFromPool(geometryDescriptor);
}

std::shared_ptr<FabricMaterial> FabricResourceManager::acquireMaterial(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const FabricMaterialInfo& materialInfo,
    const FabricFeaturesInfo& featuresInfo,
    const FabricRasterOverlaysInfo& rasterOverlaysInfo,
    int64_t tilesetId,
    const pxr::SdfPath& tilesetMaterialPath) {
    FabricMaterialDescriptor materialDescriptor(
        *_pContext, model, primitive, materialInfo, featuresInfo, rasterOverlaysInfo, tilesetMaterialPath);

    if (shouldAcquireSharedMaterial(materialDescriptor)) {
        return acquireSharedMaterial(materialInfo, materialDescriptor, tilesetId);
    }

    if (_disableMaterialPool) {
        return createMaterial(materialDescriptor);
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    return acquireMaterialFromPool(materialDescriptor);
}

std::shared_ptr<FabricTexture> FabricResourceManager::acquireTexture() {
    if (_disableTexturePool) {
        const auto name = fmt::format("/cesium_texture_{}", getNextTextureId());
        return std::make_shared<FabricTexture>(_pContext, name, -1);
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    return acquireTextureFromPool();
}

void FabricResourceManager::releaseGeometry(std::shared_ptr<FabricGeometry> pGeometry) {
    if (_disableGeometryPool) {
        return;
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    const auto pGeometryPool = getGeometryPool(*pGeometry);

    if (pGeometryPool) {
        pGeometryPool->release(std::move(pGeometry));
    }
}

void FabricResourceManager::releaseMaterial(std::shared_ptr<FabricMaterial> pMaterial) {
    if (isSharedMaterial(*pMaterial)) {
        releaseSharedMaterial(*pMaterial);
        return;
    }

    if (_disableMaterialPool) {
        return;
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    const auto pMaterialPool = getMaterialPool(*pMaterial);

    if (pMaterialPool) {
        pMaterialPool->release(std::move(pMaterial));
    }
}

void FabricResourceManager::releaseTexture(std::shared_ptr<FabricTexture> pTexture) {
    if (_disableTexturePool) {
        return;
    }

    std::scoped_lock<std::mutex> lock(_poolMutex);

    const auto pTexturePool = getTexturePool(*pTexture);

    if (pTexturePool) {
        pTexturePool->release(std::move(pTexture));
    }
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

void FabricResourceManager::updateShaderInput(
    const pxr::SdfPath& materialPath,
    const pxr::SdfPath& shaderPath,
    const pxr::TfToken& attributeName) const {
    for (const auto& pMaterialPool : _materialPools) {
        const auto& tilesetMaterialPath = pMaterialPool->getMaterialDescriptor().getTilesetMaterialPath();
        if (tilesetMaterialPath == materialPath) {
            pMaterialPool->updateShaderInput(shaderPath, attributeName);
        }
    }
}

void FabricResourceManager::clear() {
    _geometryPools.clear();
    _materialPools.clear();
    _texturePools.clear();
    _sharedMaterials.clear();
}

std::shared_ptr<FabricMaterial>
FabricResourceManager::createMaterial(const FabricMaterialDescriptor& materialDescriptor) {
    const auto pathStr = fmt::format("/cesium_material_{}", getNextMaterialId());
    const auto path = omni::fabric::Path(pathStr.c_str());
    return std::make_shared<FabricMaterial>(
        _pContext,
        path,
        materialDescriptor,
        _defaultWhiteTextureAssetPathToken,
        _defaultTransparentTextureAssetPathToken,
        _debugRandomColors,
        -1);
}

std::shared_ptr<FabricMaterial> FabricResourceManager::acquireSharedMaterial(
    const FabricMaterialInfo& materialInfo,
    const FabricMaterialDescriptor& materialDescriptor,
    int64_t tilesetId) {
    for (auto& sharedMaterial : _sharedMaterials) {
        if (sharedMaterial.materialInfo == materialInfo && sharedMaterial.tilesetId == tilesetId) {
            ++sharedMaterial.referenceCount;
            return sharedMaterial.pMaterial;
        }
    }

    const auto material = createMaterial(materialDescriptor);

    // In C++ 20 this can be emplace_back without the {}
    _sharedMaterials.push_back({
        material,
        materialInfo,
        tilesetId,
        1,
    });

    return _sharedMaterials.back().pMaterial;
}

void FabricResourceManager::releaseSharedMaterial(const FabricMaterial& material) {
    CppUtil::eraseIf(_sharedMaterials, [&material](auto& sharedMaterial) {
        if (sharedMaterial.pMaterial.get() == &material) {
            --sharedMaterial.referenceCount;
            if (sharedMaterial.referenceCount == 0) {
                return true;
            }
        }
        return false;
    });
}

bool FabricResourceManager::isSharedMaterial(const FabricMaterial& material) const {
    for (auto& sharedMaterial : _sharedMaterials) {
        if (sharedMaterial.pMaterial.get() == &material) {
            return true;
        }
    }

    return false;
}

std::shared_ptr<FabricGeometry>
FabricResourceManager::acquireGeometryFromPool(const FabricGeometryDescriptor& geometryDescriptor) {
    for (const auto& pGeometryPool : _geometryPools) {
        if (geometryDescriptor == pGeometryPool->getGeometryDescriptor()) {
            // Found a pool with the same geometry descriptor
            return pGeometryPool->acquire();
        }
    }

    auto pGeometryPool = std::make_unique<FabricGeometryPool>(
        _pContext, getNextGeometryPoolId(), geometryDescriptor, _geometryPoolInitialCapacity);

    _geometryPools.push_back(std::move(pGeometryPool));

    return _geometryPools.back()->acquire();
}

std::shared_ptr<FabricMaterial>
FabricResourceManager::acquireMaterialFromPool(const FabricMaterialDescriptor& materialDescriptor) {
    for (const auto& pMaterialPool : _materialPools) {
        if (materialDescriptor == pMaterialPool->getMaterialDescriptor()) {
            // Found a pool with the same material descriptor
            return pMaterialPool->acquire();
        }
    }

    auto pMaterialPool = std::make_unique<FabricMaterialPool>(
        _pContext,
        getNextMaterialPoolId(),
        materialDescriptor,
        _materialPoolInitialCapacity,
        _defaultWhiteTextureAssetPathToken,
        _defaultTransparentTextureAssetPathToken,
        _debugRandomColors);

    _materialPools.push_back(std::move(pMaterialPool));

    return _materialPools.back()->acquire();
}

std::shared_ptr<FabricTexture> FabricResourceManager::acquireTextureFromPool() {
    if (!_texturePools.empty()) {
        return _texturePools.front()->acquire();
    }

    auto pTexturePool =
        std::make_unique<FabricTexturePool>(_pContext, getNextTexturePoolId(), _texturePoolInitialCapacity);

    _texturePools.push_back(std::move(pTexturePool));

    return _texturePools.back()->acquire();
}

FabricGeometryPool* FabricResourceManager::getGeometryPool(const FabricGeometry& geometry) const {
    for (const auto& pGeometryPool : _geometryPools) {
        if (pGeometryPool->getPoolId() == geometry.getPoolId()) {
            return pGeometryPool.get();
        }
    }

    return nullptr;
}

FabricMaterialPool* FabricResourceManager::getMaterialPool(const FabricMaterial& material) const {
    for (const auto& pMaterialPool : _materialPools) {
        if (pMaterialPool->getPoolId() == material.getPoolId()) {
            return pMaterialPool.get();
        }
    }

    return nullptr;
}

FabricTexturePool* FabricResourceManager::getTexturePool(const FabricTexture& texture) const {
    for (const auto& pTexturePool : _texturePools) {
        if (pTexturePool->getPoolId() == texture.getPoolId()) {
            return pTexturePool.get();
        }
    }

    return nullptr;
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

int64_t FabricResourceManager::getNextGeometryPoolId() {
    return _geometryPoolId++;
}

int64_t FabricResourceManager::getNextMaterialPoolId() {
    return _materialPoolId++;
}

int64_t FabricResourceManager::getNextTexturePoolId() {
    return _texturePoolId++;
}

}; // namespace cesium::omniverse

#include "cesium/omniverse/FabricMaterialPool.h"

#include "cesium/omniverse/FabricPropertyDescriptor.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/MetadataUtil.h"

#include <spdlog/fmt/fmt.h>

namespace cesium::omniverse {

FabricMaterialPool::FabricMaterialPool(
    Context* pContext,
    int64_t poolId,
    const FabricMaterialDescriptor& materialDescriptor,
    uint64_t initialCapacity,
    const pxr::TfToken& defaultWhiteTextureAssetPathToken,
    const pxr::TfToken& defaultTransparentTextureAssetPathToken,
    bool debugRandomColors)
    : ObjectPool<FabricMaterial>()
    , _pContext(pContext)
    , _poolId(poolId)
    , _materialDescriptor(materialDescriptor)
    , _defaultWhiteTextureAssetPathToken(defaultWhiteTextureAssetPathToken)
    , _defaultTransparentTextureAssetPathToken(defaultTransparentTextureAssetPathToken)
    , _debugRandomColors(debugRandomColors) {
    setCapacity(initialCapacity);
}

const FabricMaterialDescriptor& FabricMaterialPool::getMaterialDescriptor() const {
    return _materialDescriptor;
}

int64_t FabricMaterialPool::getPoolId() const {
    return _poolId;
}

void FabricMaterialPool::updateShaderInput(const pxr::SdfPath& shaderPath, const pxr::TfToken& attributeName) {
    const auto& materials = getQueue();
    for (auto& pMaterial : materials) {
        pMaterial->updateShaderInput(FabricUtil::toFabricPath(shaderPath), FabricUtil::toFabricToken(attributeName));
    }
}

std::shared_ptr<FabricMaterial> FabricMaterialPool::createObject(uint64_t objectId) const {
    const auto contextId = _pContext->getContextId();
    const auto pathStr = fmt::format("/cesium_material_pool_{}_object_{}_context_{}", _poolId, objectId, contextId);
    const auto path = omni::fabric::Path(pathStr.c_str());
    return std::make_shared<FabricMaterial>(
        _pContext,
        path,
        _materialDescriptor,
        _defaultWhiteTextureAssetPathToken,
        _defaultTransparentTextureAssetPathToken,
        _debugRandomColors,
        _poolId);
}

void FabricMaterialPool::setActive(FabricMaterial* pMaterial, bool active) const {
    pMaterial->setActive(active);
}
}; // namespace cesium::omniverse

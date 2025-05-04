#pragma once

#include "cesium/omniverse/FabricMaterial.h"
#include "cesium/omniverse/FabricMaterialDescriptor.h"
#include "cesium/omniverse/ObjectPool.h"

#include <pxr/usd/usd/common.h>

namespace cesium::omniverse {

class FabricMaterialPool final : public ObjectPool<FabricMaterial> {
  public:
    FabricMaterialPool(
        Context* pContext,
        int64_t poolId,
        const FabricMaterialDescriptor& materialDescriptor,
        uint64_t initialCapacity,
        const omni::fabric::Token& defaultWhiteTextureAssetPathToken,
        const omni::fabric::Token& defaultTransparentTextureAssetPathToken,
        bool debugRandomColors);
    ~FabricMaterialPool() override = default;
    FabricMaterialPool(const FabricMaterialPool&) = delete;
    FabricMaterialPool& operator=(const FabricMaterialPool&) = delete;
    FabricMaterialPool(FabricMaterialPool&&) noexcept = default;
    FabricMaterialPool& operator=(FabricMaterialPool&&) noexcept = default;

    [[nodiscard]] const FabricMaterialDescriptor& getMaterialDescriptor() const;
    [[nodiscard]] int64_t getPoolId() const;

    void updateShaderInput(const pxr::SdfPath& shaderPath, const pxr::TfToken& attributeName);

  protected:
    std::shared_ptr<FabricMaterial> createObject(uint64_t objectId) const override;
    void setActive(FabricMaterial* pMaterial, bool active) const override;

  private:
    Context* _pContext;
    int64_t _poolId;
    FabricMaterialDescriptor _materialDescriptor;
    omni::fabric::Token _defaultWhiteTextureAssetPathToken;
    omni::fabric::Token _defaultTransparentTextureAssetPathToken;
    bool _debugRandomColors;
};

} // namespace cesium::omniverse

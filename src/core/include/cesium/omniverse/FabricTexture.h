#pragma once

#include <carb/RenderingTypes.h>
#include <pxr/base/tf/token.h>

#include <memory>
#include <string>

namespace omni::ui {
class DynamicTextureProvider;
}

namespace CesiumGltf {
struct ImageCesium;
}

namespace cesium::omniverse {

class Context;

enum class TransferFunction {
    LINEAR,
    SRGB,
};

class FabricTexture {
  public:
    FabricTexture(Context* pContext, const std::string& name, int64_t poolId);
    ~FabricTexture();
    FabricTexture(const FabricTexture&) = delete;
    FabricTexture& operator=(const FabricTexture&) = delete;
    FabricTexture(FabricTexture&&) noexcept = default;
    FabricTexture& operator=(FabricTexture&&) noexcept = default;

    void setImage(const CesiumGltf::ImageCesium& image, TransferFunction transferFunction);
    void setBytes(const std::vector<std::byte>& bytes, uint64_t width, uint64_t height, carb::Format format);

    void setActive(bool active);

    [[nodiscard]] const PXR_NS::TfToken& getAssetPathToken() const;
    [[nodiscard]] int64_t getPoolId() const;

  private:
    void reset();

    Context* _pContext;
    std::unique_ptr<omni::ui::DynamicTextureProvider> _pTexture;
    PXR_NS::TfToken _assetPathToken;
    int64_t _poolId;
};
} // namespace cesium::omniverse

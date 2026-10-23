#pragma once

#include <carb/RenderingTypes.h>
#include <omni/fabric/IToken.h>

#ifdef CESIUM_OMNI_MSVC
#pragma warning(push)
#pragma warning(disable : 4244 4305 4100 4201 4127 4251)
#endif

#include <pxr/base/tf/token.h>

#ifdef CESIUM_OMNI_MSVC
#pragma warning(pop)
#endif

#include <memory>
#include <string>

namespace omni::ui {
class DynamicTextureProvider;
}

namespace CesiumImage {
struct ImageAsset;
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

    void setImage(const CesiumImage::ImageAsset& image, TransferFunction transferFunction);
    void setBytes(const std::vector<std::byte>& bytes, uint64_t width, uint64_t height, carb::Format format);

    void setActive(bool active);

    [[nodiscard]] const omni::fabric::Token& getAssetPathToken() const;
    [[nodiscard]] int64_t getPoolId() const;

  private:
    void reset();

    Context* _pContext;
    std::unique_ptr<omni::ui::DynamicTextureProvider> _pTexture;
    omni::fabric::Token _assetPathToken;
    int64_t _poolId;
};
} // namespace cesium::omniverse

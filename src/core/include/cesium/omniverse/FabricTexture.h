#pragma once

#include <pxr/base/tf/token.h>
#include <pxr/usd/sdf/assetPath.h>

#include <any>
#include <memory>
#include <string>

namespace omni::ui {
class DynamicTextureProvider;
}

namespace CesiumGltf {
struct ImageCesium;
} // namespace CesiumGltf

namespace cesium::omniverse {

enum class TransferFunction {
    LINEAR,
    SRGB,
};

class FabricTexture {
  public:
    FabricTexture(const std::string& name);
    ~FabricTexture();

    void setImage(
      const CesiumGltf::ImageCesium& image,
      TransferFunction transferFunction,
      [[maybe_unused]] const std::any& rendererOptions = nullptr);

    void setActive(bool active);

    [[nodiscard]] const pxr::TfToken& getAssetPathToken() const;

  private:
    void reset();

    std::unique_ptr<omni::ui::DynamicTextureProvider> _texture;
    pxr::TfToken _assetPathToken;
};
} // namespace cesium::omniverse

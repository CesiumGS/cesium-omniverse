#pragma once

#include <pxr/base/tf/token.h>
#include <pxr/usd/sdf/assetPath.h>

#include <memory>
#include <string>

namespace omni::ui {
class DynamicTextureProvider;
}

namespace CesiumGltf {
struct ImageCesium;
} // namespace CesiumGltf

namespace cesium::omniverse {
class FabricTexture {
  public:
    FabricTexture(const std::string& name, uint64_t index);
    ~FabricTexture();

    void setImage(const CesiumGltf::ImageCesium& image);

    void setActive(bool active);

    [[nodiscard]] const pxr::TfToken& getAssetPathToken() const;
    [[nodiscard]] uint64_t getIndex() const;

  private:
    void reset();

    std::unique_ptr<omni::ui::DynamicTextureProvider> _texture;
    pxr::TfToken _assetPathToken;
    uint64_t _index;
};
} // namespace cesium::omniverse

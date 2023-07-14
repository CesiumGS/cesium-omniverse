#pragma once

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
    FabricTexture(const std::string& name);
    ~FabricTexture();

    void setImage(const CesiumGltf::ImageCesium& image);

    void setActive(bool active);

    [[nodiscard]] const pxr::SdfAssetPath& getAssetPath() const;

  private:
    void reset();

    std::unique_ptr<omni::ui::DynamicTextureProvider> _texture;
    pxr::SdfAssetPath _assetPath;
};
} // namespace cesium::omniverse

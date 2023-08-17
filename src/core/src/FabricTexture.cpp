#include "cesium/omniverse/FabricTexture.h"

#include "cesium/omniverse/FabricResourceManager.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumGltf/ImageCesium.h>
#include <carb/Types.h>
#include <omni/ui/ImageProvider/DynamicTextureProvider.h>

#include <array>

namespace cesium::omniverse {

FabricTexture::FabricTexture(const std::string& name)
    : _texture(std::make_unique<omni::ui::DynamicTextureProvider>(name))
    , _assetPathToken(UsdUtil::getDynamicTextureProviderAssetPathToken(name)) {
    const auto bytes = std::array<uint8_t, 4>{{255, 255, 255, 255}};
    const auto size = carb::Uint2{1, 1};
    _texture->setBytesData(bytes.data(), size, omni::ui::kAutoCalculateStride, carb::Format::eRGBA8_SRGB);
    FabricResourceManager::getInstance().retainToken(_assetPathToken);
    FabricResourceManager::getInstance().retainToken(pxr::TfToken(name));
}

FabricTexture::~FabricTexture() {
    int i = 0;
    (void)i;
}

void FabricTexture::setActive(bool active) {
    if (!active) {
        reset();
    }
}

const pxr::TfToken& FabricTexture::getAssetPathToken() const {
    return _assetPathToken;
}

void FabricTexture::reset() {
    // const auto bytes = std::array<uint8_t, 4>{{255, 255, 255, 255}};
    // const auto size = carb::Uint2{1, 1};
    // _texture->setBytesData(bytes.data(), size, omni::ui::kAutoCalculateStride, carb::Format::eRGBA8_SRGB);
}

void FabricTexture::setImage(const CesiumGltf::ImageCesium& image) {
    (void)image;
    // _texture->setBytesData(
    //     reinterpret_cast<const uint8_t*>(image.pixelData.data()),
    //     carb::Uint2{static_cast<uint32_t>(image.width), static_cast<uint32_t>(image.height)},
    //     omni::ui::kAutoCalculateStride,
    //     carb::Format::eRGBA8_SRGB);
}

} // namespace cesium::omniverse

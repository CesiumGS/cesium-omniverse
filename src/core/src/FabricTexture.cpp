#include "cesium/omniverse/FabricTexture.h"

#include "cesium/omniverse/LoggerSink.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumGltf/ImageCesium.h>
#include <carb/Types.h>
#include <omni/ui/ImageProvider/DynamicTextureProvider.h>

#include <array>

namespace cesium::omniverse {

namespace {
carb::Format
getCompressedImageFormat(CesiumGltf::GpuCompressedPixelFormat pixelFormat, TransferFunction transferFunction) {
    switch (pixelFormat) {
        case CesiumGltf::GpuCompressedPixelFormat::BC1_RGB:
            switch (transferFunction) {
                case TransferFunction::LINEAR:
                    return carb::Format::eBC1_RGBA_UNORM;
                case TransferFunction::SRGB:
                    return carb::Format::eBC1_RGBA_SRGB;
            }
            // Unreachable code. All enum cases are handled above.
            assert(false);
            return carb::Format::eUnknown;
        case CesiumGltf::GpuCompressedPixelFormat::BC3_RGBA:
            switch (transferFunction) {
                case TransferFunction::LINEAR:
                    return carb::Format::eBC3_RGBA_UNORM;
                case TransferFunction::SRGB:
                    return carb::Format::eBC3_RGBA_SRGB;
            }
            // Unreachable code. All enum cases are handled above.
            assert(false);
            return carb::Format::eUnknown;
        case CesiumGltf::GpuCompressedPixelFormat::BC4_R:
            return carb::Format::eBC4_R_UNORM;
        case CesiumGltf::GpuCompressedPixelFormat::BC5_RG:
            return carb::Format::eBC5_RG_UNORM;
        case CesiumGltf::GpuCompressedPixelFormat::BC7_RGBA:
            switch (transferFunction) {
                case TransferFunction::LINEAR:
                    return carb::Format::eBC7_RGBA_UNORM;
                case TransferFunction::SRGB:
                    return carb::Format::eBC7_RGBA_SRGB;
            }
            // Unreachable code. All enum cases are handled above.
            assert(false);
            return carb::Format::eUnknown;
        default:
            // Unsupported compressed texture format.
            return carb::Format::eUnknown;
    }
}

carb::Format getUncompressedPixelFormat(TransferFunction transferFunction) {
    switch (transferFunction) {
        case TransferFunction::LINEAR:
            return carb::Format::eRGBA8_UNORM;
        case TransferFunction::SRGB:
            return carb::Format::eRGBA8_SRGB;
    }

    // Unreachable code. All enum cases are handled above.
    assert(false);
    return carb::Format::eUnknown;
}

} // namespace

FabricTexture::FabricTexture(const std::string& name)
    : _texture(std::make_unique<omni::ui::DynamicTextureProvider>(name))
    , _assetPathToken(UsdUtil::getDynamicTextureProviderAssetPathToken(name)) {
    reset();
}

FabricTexture::~FabricTexture() = default;

void FabricTexture::setActive(bool active) {
    if (!active) {
        reset();
    }
}

const pxr::TfToken& FabricTexture::getAssetPathToken() const {
    return _assetPathToken;
}

void FabricTexture::reset() {
    const auto bytes = std::array<uint8_t, 4>{{255, 255, 255, 255}};
    const auto size = carb::Uint2{1, 1};
    _texture->setBytesData(bytes.data(), size, omni::ui::kAutoCalculateStride, carb::Format::eRGBA8_SRGB);
}

void FabricTexture::setImage(const CesiumGltf::ImageCesium& image, TransferFunction transferFunction) {
    carb::Format imageFormat;

    if (image.compressedPixelFormat == CesiumGltf::GpuCompressedPixelFormat::NONE) {
        imageFormat = getUncompressedPixelFormat(transferFunction);
    } else {
        imageFormat = getCompressedImageFormat(image.compressedPixelFormat, transferFunction);
    }

    if (imageFormat == carb::Format::eUnknown) {
        CESIUM_LOG_WARN("Invalid image format");
    } else {
        // As of Kit 105.1, omni::ui::kAutoCalculateStride doesn't work for compressed textures. This value somehow works.
        const auto stride = 4ULL * static_cast<uint64_t>(image.width);
        const auto data = reinterpret_cast<const uint8_t*>(image.pixelData.data());
        const auto dimensions = carb::Uint2{static_cast<uint32_t>(image.width), static_cast<uint32_t>(image.height)};

        _texture->setBytesData(data, dimensions, stride, imageFormat);
    }
}

void FabricTexture::setTexture(const TextureData& texture) {
    const auto data = reinterpret_cast<const uint8_t*>(texture.bytes.data());
    const auto dimensions = carb::Uint2{static_cast<uint32_t>(texture.width), static_cast<uint32_t>(texture.height)};
    _texture->setBytesData(data, dimensions, omni::ui::kAutoCalculateStride, texture.format);
}

} // namespace cesium::omniverse

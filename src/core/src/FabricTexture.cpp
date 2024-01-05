#include "cesium/omniverse/FabricTexture.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/FabricTextureInfo.h"
#include "cesium/omniverse/Logger.h"
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
            return carb::Format::eUnknown;
        case CesiumGltf::GpuCompressedPixelFormat::BC3_RGBA:
            switch (transferFunction) {
                case TransferFunction::LINEAR:
                    return carb::Format::eBC3_RGBA_UNORM;
                case TransferFunction::SRGB:
                    return carb::Format::eBC3_RGBA_SRGB;
            }
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
            return carb::Format::eUnknown;
        default:
            // Unsupported compressed texture format.
            return carb::Format::eUnknown;
    }
}

carb::Format
getUncompressedImageFormat(uint64_t channels, uint64_t bytesPerChannel, TransferFunction transferFunction) {
    switch (channels) {
        case 1:
            switch (bytesPerChannel) {
                case 1:
                    return carb::Format::eR8_UNORM;
                case 2:
                    return carb::Format::eR16_UNORM;
            }
            break;
        case 2:
            switch (bytesPerChannel) {
                case 1:
                    return carb::Format::eRG8_UNORM;
                case 2:
                    return carb::Format::eRG16_UNORM;
            }
            break;
        case 4:
            switch (bytesPerChannel) {
                case 1:
                    switch (transferFunction) {
                        case TransferFunction::LINEAR:
                            return carb::Format::eRGBA8_UNORM;
                        case TransferFunction::SRGB:
                            return carb::Format::eRGBA8_SRGB;
                    }
                    break;
                case 2:
                    return carb::Format::eRGBA16_UNORM;
            }
            break;
    }

    return carb::Format::eUnknown;
}

} // namespace

FabricTexture::FabricTexture(Context* pContext, const std::string& name, int64_t poolId)
    : _pContext(pContext)
    , _pTexture(std::make_unique<omni::ui::DynamicTextureProvider>(name))
    , _assetPathToken(UsdUtil::getDynamicTextureProviderAssetPathToken(name))
    , _poolId(poolId) {
    reset();
}

FabricTexture::~FabricTexture() = default;

void FabricTexture::setImage(const CesiumGltf::ImageCesium& image, TransferFunction transferFunction) {
    carb::Format imageFormat;

    const auto isCompressed = image.compressedPixelFormat != CesiumGltf::GpuCompressedPixelFormat::NONE;

    if (isCompressed) {
        imageFormat = getCompressedImageFormat(image.compressedPixelFormat, transferFunction);
    } else {
        imageFormat = getUncompressedImageFormat(
            static_cast<uint64_t>(image.channels), static_cast<uint64_t>(image.bytesPerChannel), transferFunction);
    }

    if (imageFormat == carb::Format::eUnknown) {
        _pContext->getLogger()->warn("Invalid image format");
    } else {
        // As of Kit 105.1, omni::ui::kAutoCalculateStride doesn't work for compressed textures. This value somehow works.
        const auto stride = isCompressed ? 4ULL * static_cast<uint64_t>(image.width) : omni::ui::kAutoCalculateStride;
        const auto data = reinterpret_cast<const uint8_t*>(image.pixelData.data());
        const auto dimensions = carb::Uint2{static_cast<uint32_t>(image.width), static_cast<uint32_t>(image.height)};

        _pTexture->setBytesData(data, dimensions, stride, imageFormat);
    }
}

void FabricTexture::setBytes(
    const std::vector<std::byte>& bytes,
    uint64_t width,
    uint64_t height,
    carb::Format format) {
    const auto data = reinterpret_cast<const uint8_t*>(bytes.data());
    const auto dimensions = carb::Uint2{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    _pTexture->setBytesData(data, dimensions, omni::ui::kAutoCalculateStride, format);
}

void FabricTexture::setActive(bool active) {
    if (!active) {
        reset();
    }
}

const PXR_NS::TfToken& FabricTexture::getAssetPathToken() const {
    return _assetPathToken;
}

int64_t FabricTexture::getPoolId() const {
    return _poolId;
}

void FabricTexture::reset() {
    const auto bytes = std::array<uint8_t, 4>{{255, 255, 255, 255}};
    const auto size = carb::Uint2{1, 1};
    _pTexture->setBytesData(bytes.data(), size, omni::ui::kAutoCalculateStride, carb::Format::eRGBA8_SRGB);
}

} // namespace cesium::omniverse

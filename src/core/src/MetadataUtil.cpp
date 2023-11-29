#include "cesium/omniverse/MetadataUtil.h"

#include "cesium/omniverse/DataType.h"

namespace cesium::omniverse::MetadataUtil {

namespace {
template <typename T> uint64_t indexOf(const std::vector<T>& vector, const T& value) {
    return static_cast<uint64_t>(std::distance(vector.begin(), std::find(vector.begin(), vector.end(), value)));
}
} // namespace

std::vector<MdlInternalPropertyType> getMdlInternalPropertyAttributePropertyTypes(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<MdlInternalPropertyType> mdlInternalPropertyTypes;

    forEachStyleablePropertyAttributeProperty(
        model,
        primitive,
        [&mdlInternalPropertyTypes](
            [[maybe_unused]] const std::string& propertyId,
            [[maybe_unused]] auto propertyAttributePropertyView,
            auto styleableProperty) {
            constexpr auto type = decltype(styleableProperty)::Type;
            mdlInternalPropertyTypes.push_back(getMdlInternalPropertyType<type>());
        });

    std::sort(mdlInternalPropertyTypes.begin(), mdlInternalPropertyTypes.end());

    return mdlInternalPropertyTypes;
}

std::vector<MdlInternalPropertyType>
getMdlInternalPropertyTexturePropertyTypes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<MdlInternalPropertyType> mdlInternalPropertyTypes;

    forEachStyleablePropertyTextureProperty(
        model,
        primitive,
        [&mdlInternalPropertyTypes](
            [[maybe_unused]] const std::string& propertyId,
            [[maybe_unused]] auto propertyTexturePropertyView,
            auto styleableProperty) {
            constexpr auto type = decltype(styleableProperty)::Type;
            mdlInternalPropertyTypes.push_back(getMdlInternalPropertyType<type>());
        });

    std::sort(mdlInternalPropertyTypes.begin(), mdlInternalPropertyTypes.end());

    return mdlInternalPropertyTypes;
}

std::vector<MdlInternalPropertyType>
getMdlInternalPropertyTablePropertyTypes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<MdlInternalPropertyType> mdlInternalPropertyTypes;

    forEachStyleablePropertyTableProperty(
        model,
        primitive,
        [&mdlInternalPropertyTypes](
            [[maybe_unused]] const std::string& propertyId,
            [[maybe_unused]] auto propertyTexturePropertyView,
            auto styleableProperty) {
            constexpr auto type = decltype(styleableProperty)::Type;
            mdlInternalPropertyTypes.push_back(getMdlInternalPropertyType<type>());
        });

    std::sort(mdlInternalPropertyTypes.begin(), mdlInternalPropertyTypes.end());

    return mdlInternalPropertyTypes;
}

std::vector<const CesiumGltf::ImageCesium*>
getPropertyTextureImages(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<const CesiumGltf::ImageCesium*> images;

    forEachStyleablePropertyTextureProperty(
        model,
        primitive,
        [&images](
            [[maybe_unused]] const std::string& propertyId,
            auto propertyTexturePropertyView,
            [[maybe_unused]] auto styleableProperty) {
            const auto pImage = propertyTexturePropertyView.getImage();
            assert(pImage);

            const auto imageIndex = indexOf(images, pImage);

            if (imageIndex == images.size()) {
                images.push_back(pImage);
            }
        });

    return images;
}

std::unordered_map<uint64_t, uint64_t>
getPropertyTextureIndexMapping(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<const CesiumGltf::ImageCesium*> images;
    std::unordered_map<uint64_t, uint64_t> propertyTextureIndexMapping;

    forEachStyleablePropertyTextureProperty(
        model,
        primitive,
        [&images, &propertyTextureIndexMapping](
            [[maybe_unused]] const std::string& propertyId,
            auto propertyTexturePropertyView,
            [[maybe_unused]] auto styleableProperty) {
            const auto pImage = propertyTexturePropertyView.getImage();
            assert(pImage);

            const auto imageIndex = indexOf(images, pImage);

            if (imageIndex == images.size()) {
                images.push_back(pImage);
            }

            const auto textureIndex = styleableProperty.textureIndex;
            propertyTextureIndexMapping[textureIndex] = imageIndex;
        });

    return propertyTextureIndexMapping;
}

std::vector<TextureData>
encodePropertyTables(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<TextureData> textures;

    forEachStyleablePropertyTableProperty(
        model,
        primitive,
        [&textures](
            [[maybe_unused]] const std::string& propertyId, auto propertyTablePropertyView, auto styleableProperty) {
            constexpr auto type = decltype(styleableProperty)::Type;
            constexpr auto textureType = getPropertyTableTextureType<type>();
            constexpr auto textureFormat = getTextureFormat<textureType>();
            using TextureType = GetNativeType<textureType>;
            using TextureComponentType = GetNativeType<getComponentType<textureType>()>;

            // The texture type should always be the same or larger type
            static_assert(getComponentCount<type>() <= getComponentCount<textureType>());

            // Matrix packing not implemented yet
            static_assert(!isMatrix<type>());

            const auto size = static_cast<uint64_t>(propertyTablePropertyView.size());
            assert(size > 0);

            constexpr uint64_t maximumTextureWidth = 4096;
            const auto width = glm::min(maximumTextureWidth, size);
            const auto height = ((size - 1) / maximumTextureWidth) + 1;

            constexpr auto texelByteLength = getByteLength<textureType>();
            const auto textureByteLength = width * height * texelByteLength;

            std::vector<std::byte> texelBytes(textureByteLength, std::byte(0));
            gsl::span<TextureType> texelValues(
                reinterpret_cast<TextureType*>(texelBytes.data()), texelBytes.size() / texelByteLength);

            for (uint64_t i = 0; i < size; i++) {
                auto& texelValue = texelValues[i];
                const auto& rawValue = propertyTablePropertyView.getRaw(i);

                if constexpr (isVector<type>()) {
                    for (uint64_t j = 0; j < getComponentCount<type>(); j++) {
                        texelValue[j] = static_cast<TextureComponentType>(rawValue[j]);
                    }
                } else {
                    texelValue = static_cast<TextureType>(rawValue);
                }
            }

            textures.emplace_back(TextureData{
                std::move(texelBytes),
                width,
                height,
                textureFormat,
            });
        });

    return textures;
}

} // namespace cesium::omniverse::MetadataUtil

#include "cesium/omniverse/MetadataUtil.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/DataType.h"
#include "cesium/omniverse/FabricPropertyDescriptor.h"
#include "cesium/omniverse/FabricTextureData.h"

namespace cesium::omniverse::MetadataUtil {

std::tuple<std::vector<FabricPropertyDescriptor>, std::map<std::string, std::string>> getStyleableProperties(
    const Context& context,
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<FabricPropertyDescriptor> properties;
    std::map<std::string, std::string> unsupportedPropertyWarnings;

    forEachStyleablePropertyAttributeProperty(
        context,
        model,
        primitive,
        [&properties](
            const std::string& propertyId,
            [[maybe_unused]] const auto& propertyAttributePropertyView,
            const auto& property) {
            constexpr auto type = std::decay_t<decltype(property)>::Type;

            // In C++ 20 this can be emplace_back without the {}
            properties.push_back({
                FabricPropertyStorageType::ATTRIBUTE,
                DataTypeUtil::getMdlInternalPropertyType<type>(),
                propertyId,
                0, // featureIdSetIndex not relevant for property attributes
            });
        },
        [&unsupportedPropertyWarnings](const std::string& propertyId, const std::string& warning) {
            unsupportedPropertyWarnings.insert({propertyId, warning});
        });

    forEachStyleablePropertyTextureProperty(
        context,
        model,
        primitive,
        [&properties](
            const std::string& propertyId,
            [[maybe_unused]] const auto& propertyTexturePropertyView,
            const auto& property) {
            constexpr auto type = std::decay_t<decltype(property)>::Type;

            // In C++ 20 this can be emplace_back without the {}
            properties.push_back({
                FabricPropertyStorageType::TEXTURE,
                DataTypeUtil::getMdlInternalPropertyType<type>(),
                propertyId,
                0, // featureIdSetIndex not relevant for property textures
            });
        },
        [&unsupportedPropertyWarnings](const std::string& propertyId, const std::string& warning) {
            unsupportedPropertyWarnings.insert({propertyId, warning});
        });

    forEachStyleablePropertyTableProperty(
        context,
        model,
        primitive,
        [&properties](
            const std::string& propertyId,
            [[maybe_unused]] const auto& propertyTablePropertyView,
            const auto& property) {
            constexpr auto type = std::decay_t<decltype(property)>::Type;

            // In C++ 20 this can be emplace_back without the {}
            properties.push_back({
                FabricPropertyStorageType::TABLE,
                DataTypeUtil::getMdlInternalPropertyType<type>(),
                propertyId,
                property.featureIdSetIndex,
            });
        },
        [&unsupportedPropertyWarnings](const std::string& propertyId, const std::string& warning) {
            unsupportedPropertyWarnings.insert({propertyId, warning});
        });

    // Sorting is important for checking FabricMaterialDescriptor equality
    CppUtil::sort(properties, [](const auto& lhs, const auto& rhs) { return lhs.propertyId > rhs.propertyId; });

    return {std::move(properties), std::move(unsupportedPropertyWarnings)};
}

std::vector<const CesiumGltf::ImageCesium*> getPropertyTextureImages(
    const Context& context,
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<const CesiumGltf::ImageCesium*> images;

    forEachStyleablePropertyTextureProperty(
        context,
        model,
        primitive,
        [&images](
            [[maybe_unused]] const std::string& propertyId,
            const auto& propertyTexturePropertyView,
            [[maybe_unused]] const auto& property) {
            const auto pImage = propertyTexturePropertyView.getImage();
            assert(pImage); // Shouldn't have gotten this far if image is invalid

            const auto imageIndex = CppUtil::indexOf(images, pImage);
            if (imageIndex == images.size()) {
                images.push_back(pImage);
            }
        },
        []([[maybe_unused]] const std::string& propertyId, [[maybe_unused]] const std::string& warning) {});

    return images;
}

std::unordered_map<uint64_t, uint64_t> getPropertyTextureIndexMapping(
    const Context& context,
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<const CesiumGltf::ImageCesium*> images;
    std::unordered_map<uint64_t, uint64_t> propertyTextureIndexMapping;

    forEachStyleablePropertyTextureProperty(
        context,
        model,
        primitive,
        [&images, &propertyTextureIndexMapping](
            [[maybe_unused]] const std::string& propertyId,
            const auto& propertyTexturePropertyView,
            [[maybe_unused]] const auto& property) {
            const auto pImage = propertyTexturePropertyView.getImage();
            assert(pImage); // Shouldn't have gotten this far if image is invalid

            const auto imageIndex = CppUtil::indexOf(images, pImage);
            if (imageIndex == images.size()) {
                images.push_back(pImage);
            }

            const auto textureIndex = property.textureIndex;
            propertyTextureIndexMapping[textureIndex] = imageIndex;
        },
        []([[maybe_unused]] const std::string& propertyId, [[maybe_unused]] const std::string& warning) {});

    return propertyTextureIndexMapping;
}

std::vector<FabricTextureData> encodePropertyTables(
    const Context& context,
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<FabricTextureData> textures;

    forEachStyleablePropertyTableProperty(
        context,
        model,
        primitive,
        [&textures](
            [[maybe_unused]] const std::string& propertyId,
            const auto& propertyTablePropertyView,
            const auto& property) {
            constexpr auto type = std::decay_t<decltype(property)>::Type;
            constexpr auto textureType = DataTypeUtil::getPropertyTableTextureType<type>();
            constexpr auto textureFormat = DataTypeUtil::getTextureFormat<textureType>();
            using TextureType = DataTypeUtil::GetNativeType<textureType>;
            using TextureComponentType = DataTypeUtil::GetNativeType<DataTypeUtil::getComponentType<textureType>()>;

            // The texture type should always be the same or larger type
            static_assert(DataTypeUtil::getComponentCount<type>() <= DataTypeUtil::getComponentCount<textureType>());

            // Matrix packing not implemented yet
            static_assert(!DataTypeUtil::isMatrix<type>());

            const auto size = static_cast<uint64_t>(propertyTablePropertyView.size());
            assert(size > 0);

            constexpr uint64_t maximumTextureWidth = 4096;
            const auto width = glm::min(maximumTextureWidth, size);
            const auto height = ((size - 1) / maximumTextureWidth) + 1;

            constexpr auto texelByteLength = DataTypeUtil::getByteLength<textureType>();
            const auto textureByteLength = width * height * texelByteLength;

            std::vector<std::byte> texelBytes(textureByteLength, std::byte(0));
            gsl::span<TextureType> texelValues(
                reinterpret_cast<TextureType*>(texelBytes.data()), texelBytes.size() / texelByteLength);

            for (uint64_t i = 0; i < size; ++i) {
                auto& texelValue = texelValues[i];
                const auto& rawValue = propertyTablePropertyView.getRaw(static_cast<int64_t>(i));

                if constexpr (DataTypeUtil::isVector<type>()) {
                    for (uint64_t j = 0; j < DataTypeUtil::getComponentCount<type>(); ++j) {
                        texelValue[j] = static_cast<TextureComponentType>(rawValue[j]);
                    }
                } else {
                    texelValue = static_cast<TextureType>(rawValue);
                }
            }

            // In C++ 20 this can be push_back without the {}
            textures.push_back({
                std::move(texelBytes),
                width,
                height,
                textureFormat,
            });
        },
        []([[maybe_unused]] const std::string& propertyId, [[maybe_unused]] const std::string& warning) {});

    return textures;
}

uint64_t getPropertyTableTextureCount(
    const Context& context,
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive) {
    uint64_t count = 0;

    forEachStyleablePropertyTableProperty(
        context,
        model,
        primitive,
        [&count](
            [[maybe_unused]] const std::string& propertyId,
            [[maybe_unused]] const auto& propertyTablePropertyView,
            [[maybe_unused]] const auto& property) { ++count; },
        []([[maybe_unused]] const std::string& propertyId, [[maybe_unused]] const std::string& warning) {});

    return count;
}

} // namespace cesium::omniverse::MetadataUtil

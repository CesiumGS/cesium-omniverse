#include "cesium/omniverse/MetadataUtil.h"

#include "cesium/omniverse/DataType.h"

namespace cesium::omniverse::MetadataUtil {

namespace {
template <typename T> uint64_t indexOf(const std::vector<T>& vector, const T& value) {
    return static_cast<uint64_t>(std::distance(vector.begin(), std::find(vector.begin(), vector.end(), value)));
}
} // namespace

std::vector<MetadataUtil::PropertyDefinition>
getStyleableProperties(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<MetadataUtil::PropertyDefinition> properties;

    forEachStyleablePropertyAttributeProperty(
        model,
        primitive,
        [&properties](
            const std::string& propertyId,
            [[maybe_unused]] const auto& propertyAttributePropertyView,
            const auto& property) {
            constexpr auto type = std::decay_t<decltype(property)>::Type;
            properties.emplace_back(MetadataUtil::PropertyDefinition{
                PropertyStorageType::ATTRIBUTE,
                getMdlInternalPropertyType<type>(),
                propertyId,
                0, // featureIdSetIndex not relevant for property attributes
            });
        });

    forEachStyleablePropertyTextureProperty(
        model,
        primitive,
        [&properties](
            const std::string& propertyId,
            [[maybe_unused]] const auto& propertyTexturePropertyView,
            const auto& property) {
            constexpr auto type = std::decay_t<decltype(property)>::Type;
            properties.emplace_back(MetadataUtil::PropertyDefinition{
                PropertyStorageType::TEXTURE,
                getMdlInternalPropertyType<type>(),
                propertyId,
                0, // featureIdSetIndex not relevant for property textures
            });
        });

    forEachStyleablePropertyTableProperty(
        model,
        primitive,
        [&properties](
            const std::string& propertyId,
            [[maybe_unused]] const auto& propertyTablePropertyView,
            const auto& property) {
            constexpr auto type = std::decay_t<decltype(property)>::Type;
            properties.emplace_back(MetadataUtil::PropertyDefinition{
                PropertyStorageType::TABLE,
                getMdlInternalPropertyType<type>(),
                propertyId,
                property.featureIdSetIndex,
            });
        });

    // Sorting is important for checking FabricMaterialDefinition equality
    std::sort(
        properties.begin(), properties.end(), [](const PropertyDefinition& lhs, const PropertyDefinition& rhs) -> bool {
            return lhs.propertyId > rhs.propertyId;
        });

    return properties;
}

std::vector<const CesiumGltf::ImageCesium*>
getPropertyTextureImages(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<const CesiumGltf::ImageCesium*> images;

    forEachStyleablePropertyTextureProperty(
        model,
        primitive,
        [&images](
            [[maybe_unused]] const std::string& propertyId,
            const auto& propertyTexturePropertyView,
            [[maybe_unused]] const auto& property) {
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
            const auto& propertyTexturePropertyView,
            [[maybe_unused]] const auto& property) {
            const auto pImage = propertyTexturePropertyView.getImage();
            assert(pImage);

            const auto imageIndex = indexOf(images, pImage);

            if (imageIndex == images.size()) {
                images.push_back(pImage);
            }

            const auto textureIndex = property.textureIndex;
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
            [[maybe_unused]] const std::string& propertyId,
            const auto& propertyTablePropertyView,
            const auto& property) {
            constexpr auto type = std::decay_t<decltype(property)>::Type;
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

uint64_t getPropertyTableTextureCount(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    uint64_t count = 0;

    forEachStyleablePropertyTableProperty(
        model,
        primitive,
        [&count](
            [[maybe_unused]] const std::string& propertyId,
            [[maybe_unused]] const auto& propertyTablePropertyView,
            [[maybe_unused]] const auto& property) { count++; });

    return count;
}

// In C++ 20 we can use the default equality comparison (= default)
bool PropertyDefinition::operator==(const PropertyDefinition& other) const {
    return storageType == other.storageType && type == other.type && propertyId == other.propertyId &&
           featureIdSetIndex == other.featureIdSetIndex;
}

} // namespace cesium::omniverse::MetadataUtil

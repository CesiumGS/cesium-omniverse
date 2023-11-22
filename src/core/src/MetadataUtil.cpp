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
            [[maybe_unused]] auto propertyAttributeProperty,
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
            [[maybe_unused]] auto propertyTextureProperty,
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
            [[maybe_unused]] auto propertyTextureProperty,
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
            auto propertyTextureProperty,
            auto propertyTexturePropertyView,
            [[maybe_unused]] auto styleableProperty) {
            const auto pImage = propertyTexturePropertyView.getImage();
            assert(pImage);

            const auto imageIndex = indexOf(images, pImage);

            if (imageIndex == images.size()) {
                images.push_back(pImage);
            }

            const auto textureIndex = static_cast<uint64_t>(propertyTextureProperty.index);
            propertyTextureIndexMapping[textureIndex] = imageIndex;
        });

    return propertyTextureIndexMapping;
}

} // namespace cesium::omniverse::MetadataUtil

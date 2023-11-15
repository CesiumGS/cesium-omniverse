#include "cesium/omniverse/MetadataUtil.h"

namespace cesium::omniverse::MetadataUtil {
std::vector<DataType>
getMdlPropertyAttributeTypes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<DataType> mdlTypes;

    forEachStyleablePropertyAttributeProperty(
        model,
        primitive,
        [&mdlTypes](
            [[maybe_unused]] const std::string& propertyId,
            [[maybe_unused]] const CesiumGltf::Schema& schema,
            [[maybe_unused]] const CesiumGltf::Class& classDefinition,
            [[maybe_unused]] const CesiumGltf::ClassProperty& classProperty,
            [[maybe_unused]] const CesiumGltf::PropertyAttribute& propertyAttribute,
            [[maybe_unused]] const CesiumGltf::PropertyAttributeProperty& propertyAttributeProperty,
            [[maybe_unused]] const CesiumGltf::PropertyAttributeView& propertyTextureView,
            [[maybe_unused]] auto propertyTexturePropertyView,
            auto styleableProperty) { mdlTypes.push_back(getMdlPropertyType(styleableProperty.type)); });

    std::sort(mdlTypes.begin(), mdlTypes.end());

    return mdlTypes;
}

std::vector<DataType>
getMdlPropertyTextureTypes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<DataType> mdlTypes;

    forEachStyleablePropertyTextureProperty(
        model,
        primitive,
        [&mdlTypes](
            [[maybe_unused]] const std::string& propertyId,
            [[maybe_unused]] const CesiumGltf::Schema& schema,
            [[maybe_unused]] const CesiumGltf::Class& classDefinition,
            [[maybe_unused]] const CesiumGltf::ClassProperty& classProperty,
            [[maybe_unused]] const CesiumGltf::PropertyTexture& propertyTexture,
            [[maybe_unused]] const CesiumGltf::PropertyTextureProperty& propertyTextureProperty,
            [[maybe_unused]] const CesiumGltf::PropertyTextureView& propertyTextureView,
            [[maybe_unused]] auto propertyTexturePropertyView,
            auto styleableProperty) { mdlTypes.push_back(getMdlPropertyType(styleableProperty.type)); });

    std::sort(mdlTypes.begin(), mdlTypes.end());

    return mdlTypes;
}

std::vector<const CesiumGltf::ImageCesium*>
getImagesReferencedByPropertyTextures(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<const CesiumGltf::ImageCesium*> images;

    forEachStyleablePropertyTextureProperty(
        model,
        primitive,
        [&images](
            [[maybe_unused]] const std::string& propertyId,
            [[maybe_unused]] const CesiumGltf::Schema& schema,
            [[maybe_unused]] const CesiumGltf::Class& classDefinition,
            [[maybe_unused]] const CesiumGltf::ClassProperty& classProperty,
            [[maybe_unused]] const CesiumGltf::PropertyTexture& propertyTexture,
            [[maybe_unused]] const CesiumGltf::PropertyTextureProperty& propertyTextureProperty,
            [[maybe_unused]] const CesiumGltf::PropertyTextureView& propertyTextureView,
            auto propertyTexturePropertyView,
            [[maybe_unused]] auto styleableProperty) {
            const auto pImage = propertyTexturePropertyView.getImage();
            assert(pImage);
            images.push_back(pImage);
        });

    std::sort(images.begin(), images.end());
    const auto endNew = std::unique(images.begin(), images.end());
    images.resize(static_cast<size_t>(std::distance(images.begin(), endNew)));

    return images;
}

} // namespace cesium::omniverse::MetadataUtil

#include "cesium/omniverse/MetadataUtil.h"

namespace cesium::omniverse::MetadataUtil {
std::vector<DataType>
getMdlPropertyAttributeTypes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<DataType> mdlTypes;

    forEachStyleablePropertyAttributeProperty(
        model, primitive, [&mdlTypes]([[maybe_unused]] auto propertyAttributePropertyView, auto styleableProperty) {
            constexpr auto Type = decltype(styleableProperty)::Type;
            mdlTypes.push_back(GetMdlShaderType<Type>::Type);
        });

    std::sort(mdlTypes.begin(), mdlTypes.end());

    return mdlTypes;
}

std::vector<DataType>
getMdlPropertyTextureTypes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<DataType> mdlTypes;

    forEachStyleablePropertyTextureProperty(
        model, primitive, [&mdlTypes]([[maybe_unused]] auto propertyTexturePropertyView, auto styleableProperty) {
            constexpr auto Type = decltype(styleableProperty)::Type;
            mdlTypes.push_back(GetMdlShaderType<Type>::Type);
        });

    std::sort(mdlTypes.begin(), mdlTypes.end());

    return mdlTypes;
}

std::vector<const CesiumGltf::ImageCesium*>
getPropertyTextureImages(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    std::vector<const CesiumGltf::ImageCesium*> images;

    forEachStyleablePropertyTextureProperty(
        model, primitive, [&images](auto propertyTexturePropertyView, [[maybe_unused]] auto styleableProperty) {
            const auto pImage = propertyTexturePropertyView.getImage();
            assert(pImage);
            images.push_back(pImage);
        });

    // Remove duplicates
    std::sort(images.begin(), images.end());
    const auto endNew = std::unique(images.begin(), images.end());
    images.resize(static_cast<size_t>(std::distance(images.begin(), endNew)));

    return images;
}

} // namespace cesium::omniverse::MetadataUtil

#include "cesium/omniverse/MetadataUtil.h"

#include "cesium/omniverse/DataType.h"

namespace cesium::omniverse::MetadataUtil {

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

} // namespace cesium::omniverse::MetadataUtil

#include "cesium/omniverse/FabricGltfUtil.h"

#include "cesium/omniverse/GltfUtil.h"

#include <CesiumGltf/AccessorView.h>

namespace cesium::omniverse::FabricGltfUtil {

std::set<FabricVertexAttributeDescriptor>
getCustomVertexAttributes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    constexpr std::array<std::string_view, 8> knownSemantics = {{
        "POSITION",
        "NORMAL",
        "TANGENT",
        "TEXCOORD",
        "COLOR",
        "JOINTS",
        "WEIGHTS",
        "_CESIUMOVERLAY",
    }};

    std::set<FabricVertexAttributeDescriptor> customVertexAttributes;

    for (const auto& attribute : primitive.attributes) {
        const auto& attributeName = attribute.first;
        const auto [semantic, setIndex] = GltfUtil::parseAttributeName(attributeName);
        if (CppUtil::contains(knownSemantics, semantic)) {
            continue;
        }

        auto pAccessor = model.getSafe(&model.accessors, static_cast<int32_t>(attribute.second));
        if (!pAccessor) {
            continue;
        }

        const auto valid = createAccessorView(model, *pAccessor, [](const auto& accessorView) {
            return accessorView.status() == CesiumGltf::AccessorViewStatus::Valid;
        });

        if (!valid) {
            continue;
        }

        const auto type = GltfUtil::getVertexAttributeTypeFromGltf(*pAccessor);

        if (!type.has_value()) {
            continue;
        }

        const auto fabricAttributeNameStr = fmt::format("primvars:{}", attributeName);
        const auto fabricInterpolationNameStr = fmt::format("primvars:{}:interpolation", attributeName);
        const auto fabricAttributeName = omni::fabric::Token::createImmortal(fabricAttributeNameStr.c_str());
        const auto fabricInterpolationName = omni::fabric::Token::createImmortal(fabricInterpolationNameStr.c_str());

        customVertexAttributes.emplace(type.value(), fabricAttributeName, fabricInterpolationName, attributeName);
    }

    return customVertexAttributes;
}

} // namespace cesium::omniverse::FabricGltfUtil
